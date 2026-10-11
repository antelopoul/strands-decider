"""StrandsDeciderModel.load: a complete checkpoint loads unchanged. An incomplete or ambiguous one
is refused before the torso loads, and the head pickle is read with weights_only.

CPU only: a tiny random Qwen3 torso stands in for the base model, so nothing is
downloaded."""

import copy
import os
import pickle
import shutil

import pytest
import torch

from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel


class _NotATensor:
    """A global that torch's weights_only unpickler refuses."""


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(["<pad>", "<eos>", "<unk>", "a", "b", "c"])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>",
                                   unk_token="<unk>")


@pytest.fixture
def ckpt(tmp_path, monkeypatch):
    """(checkpoint dir, the saved model, the torso loads that StrandsDeciderModel.load made)."""
    from transformers import Qwen3Config, Qwen3Model

    base = Qwen3Model(Qwen3Config(vocab_size=6, hidden_size=32, intermediate_size=64,
                                  num_hidden_layers=1, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=8))
    loads = []

    def load_torso(config, device_map, attn_implementation):
        loads.append(config.base_model)
        return copy.deepcopy(base)

    monkeypatch.setattr(StrandsDeciderModel, "_load_torso", staticmethod(load_torso))
    cfg = StrandsDeciderConfig(base_model="tiny", head_type="pointer", pointer_dim=16, lora_r=2,
                       torch_dtype="float32")
    model = StrandsDeciderModel(cfg, copy.deepcopy(base), _tokenizer())
    model.attach_lora()
    with torch.no_grad():  # lora_B starts at zero: make the adapter change the torso
        for n, p in model.torso.named_parameters():
            if "lora_B" in n:
                p.normal_()
    path = tmp_path / "ckpt"
    model.save_pretrained(str(path))
    return path, model, loads


def _assert_torso_state_is_preserved(source, continued):
    source_state = source.torso.state_dict()
    continued_state = continued.torso.state_dict()
    assert source_state.keys() == continued_state.keys()
    assert all(torch.equal(source_state[k], continued_state[k]) for k in source_state)


def test_init_from_builds_requested_pointer_head_and_forward_path(ckpt):
    from strands_decider.modeling import PointerHead
    from strands_decider.train import TrainConfig, _build_model

    path, source, _ = ckpt
    model = _build_model(TrainConfig(
        init_from=str(path),
        head_type="pointer",
        num_slots=11,
        pointer_dim=8,
        head_dropout=0.1,
    ))

    assert isinstance(model.head, PointerHead)
    assert model.config.head_type == "pointer"
    assert model.config.num_slots == 11
    assert model.config.pointer_dim == 8
    assert model.head.q.out_features == 8
    _assert_torso_state_is_preserved(source, model)
    assert all(not p.requires_grad for p in model.torso.parameters())

    out = model(
        input_ids=torch.tensor([[3, 4, 5]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        n_slots=torch.tensor([2]),
        opt_idx=torch.tensor([[0, 1]]),
    )
    assert out["logits"].shape == (1, 2)


def test_init_from_builds_requested_slot_head_and_forward_path(ckpt):
    from strands_decider.modeling import SlotHead
    from strands_decider.train import TrainConfig, _build_model

    path, source, _ = ckpt
    model = _build_model(TrainConfig(
        init_from=str(path),
        head_type="slot",
        num_slots=7,
        head_hidden=12,
        head_dropout=0.2,
    ))

    assert isinstance(model.head, SlotHead)
    assert model.config.head_type == "slot"
    assert model.config.num_slots == 7
    assert model.config.head_hidden == 12
    assert model.config.head_dropout == 0.2
    assert model.head.proj[0].out_features == 12
    _assert_torso_state_is_preserved(source, model)
    assert all(not p.requires_grad for p in model.torso.parameters())

    out = model(
        input_ids=torch.tensor([[3, 4, 5]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        n_slots=torch.tensor([4]),
    )
    assert out["logits"].shape == (1, 7)


def test_lm_head_init_rejects_pointer_and_hidden_slot_heads(ckpt):
    from strands_decider.modeling import StrandsDeciderConfig
    from strands_decider.train import TrainConfig, _build_model

    path, model, loads = ckpt
    message = (
        "head_init='lm_head' requires a linear slot head "
        "(head_type='slot' and head_hidden=0); got head_type='pointer', head_hidden=0"
    )
    with pytest.raises(ValueError, match="head_init='lm_head' requires a linear slot head") as exc:
        _build_model(TrainConfig(init_from=str(path), head_type="pointer", head_init="lm_head"))
    assert str(exc.value) == message
    assert loads == []

    with pytest.raises(ValueError, match="head_init='lm_head' requires a linear slot head") as exc:
        model.init_head_from_lm_head()
    assert str(exc.value) == message

    hidden_config = StrandsDeciderConfig(head_type="slot", head_hidden=8)
    with pytest.raises(ValueError, match="head_init='lm_head' requires a linear slot head") as exc:
        hidden_config.validate_lm_head_init()
    assert str(exc.value).endswith("got head_type='slot', head_hidden=8")


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_a_complete_checkpoint_and_its_safetensors_head_load_unchanged(ckpt):
    path, model, loads = ckpt
    assert _same(StrandsDeciderModel.load(str(path)), model)
    from safetensors.torch import save_file

    save_file(torch.load(path / "slot_head.pt", weights_only=True), str(path / "head.safetensors"))
    os.remove(path / "slot_head.pt")
    assert _same(StrandsDeciderModel.load(str(path)), model)
    assert loads == ["tiny", "tiny"]


def test_a_missing_adapter_is_refused_before_the_torso_loads(ckpt):
    path, _, loads = ckpt
    shutil.rmtree(path / "lora")
    with pytest.raises(FileNotFoundError, match="lora"):
        StrandsDeciderModel.load(str(path))
    assert loads == []


def test_two_head_files_are_refused_before_the_torso_loads(ckpt):
    path, model, loads = ckpt
    from safetensors.torch import save_file

    save_file({k: torch.zeros_like(v) for k, v in model.head.state_dict().items()},
              str(path / "head.safetensors"))
    with pytest.raises(ValueError, match="both head"):
        StrandsDeciderModel.load(str(path))
    assert loads == []


def test_the_head_pickle_is_read_with_weights_only(ckpt, monkeypatch):
    path, model, loads = ckpt
    torch.save(dict(model.head.state_dict(), extra=_NotATensor()), path / "slot_head.pt")
    # Makes torch.load unpickle anything unless the call itself sets weights_only=True.
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        StrandsDeciderModel.load(str(path))
    assert loads == []


def test_a_hub_repo_id_loads_from_the_downloaded_snapshot(ckpt, monkeypatch):
    """A path that is not a local directory is a Hub repo id: snapshot_download gives the
    folder. No network here: the download is stubbed, and a failure names both readings."""
    import strands_decider.modeling as modeling

    path, model, loads = ckpt
    asked = []

    def download(repo_id):
        asked.append(repo_id)
        return str(path)

    monkeypatch.setattr(modeling, "snapshot_download", download)
    assert _same(StrandsDeciderModel.load("org/hobson-test"), model)
    assert asked == ["org/hobson-test"] and loads == ["tiny"]

    def refuse(repo_id):
        raise OSError("offline")

    monkeypatch.setattr(modeling, "snapshot_download", refuse)
    with pytest.raises(FileNotFoundError, match="not a local directory, and could not be "
                                                "downloaded from the Hugging Face Hub: OSError: offline"):
        StrandsDeciderModel.load(str(path / "typo"))
    assert loads == ["tiny"]


def test_a_local_directory_named_like_a_repo_id_is_not_downloaded(ckpt, monkeypatch, tmp_path):
    import strands_decider.modeling as modeling

    path, model, loads = ckpt
    local = tmp_path / "org" / "hobson-test"
    shutil.copytree(path, local)
    monkeypatch.setattr(modeling, "snapshot_download",
                        lambda repo_id: pytest.fail(f"downloaded {repo_id}"))
    monkeypatch.chdir(tmp_path)
    assert _same(StrandsDeciderModel.load("org/hobson-test"), model)
    assert loads == ["tiny"]


def test_hobson_info_accepts_a_repo_id(ckpt, monkeypatch):
    from typer.testing import CliRunner

    import strands_decider.modeling as modeling
    from strands_decider.cli import app

    path, _, loads = ckpt
    monkeypatch.setattr(modeling, "snapshot_download", lambda repo_id: str(path))
    res = CliRunner().invoke(app, ["info", "org/hobson-test"])
    assert res.exit_code == 0, res.output
    assert '"base_model": "tiny"' in res.output and loads == []


def test_calibration_refuses_a_repo_id_before_the_fit(ckpt, monkeypatch):
    """calibrate_checkpoint writes hobson_config.json into the checkpoint at the end, so a
    downloaded snapshot would lose the fit. It is refused before anything loads."""
    from strands_decider.evaluate import calibrate_checkpoint

    _, _, loads = ckpt
    with pytest.raises(FileNotFoundError, match="needs a local directory"):
        calibrate_checkpoint("org/hobson-test", [])
    assert loads == []
