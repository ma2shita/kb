"""Tests for kb.embed — serialization and embedding."""

import struct
from unittest.mock import MagicMock

import pytest

from kb.config import Config
from kb.embed import deserialize_f32, embed_batch, local_embed_batch, serialize_f32


@pytest.fixture
def mock_local_model(monkeypatch):
    model = MagicMock()
    model.prompts = {}
    embedding = MagicMock()
    embedding.tolist.return_value = [0.1, 0.2, 0.3]
    model.encode.side_effect = lambda texts, **kwargs: [embedding for _ in texts]
    monkeypatch.setattr("kb.embed._get_embed_model", lambda name: model)
    return model


class TestSerializeF32:
    def test_roundtrip(self):
        vec = [1.0, 2.5, -3.14, 0.0]
        serialized = serialize_f32(vec)
        assert isinstance(serialized, bytes)
        assert len(serialized) == 4 * len(vec)
        unpacked = list(struct.unpack(f"{len(vec)}f", serialized))
        for a, b in zip(vec, unpacked):
            assert abs(a - b) < 1e-5

    def test_empty_vector(self):
        assert serialize_f32([]) == b""

    def test_single_element(self):
        serialized = serialize_f32([42.0])
        assert len(serialized) == 4


class TestDeserializeF32:
    def test_roundtrip(self):
        vec = [1.0, 2.5, -3.14, 0.0]
        blob = serialize_f32(vec)
        result = deserialize_f32(blob)
        assert len(result) == len(vec)
        for a, b in zip(vec, result):
            assert abs(a - b) < 1e-5

    def test_empty(self):
        assert deserialize_f32(b"") == []

    def test_single_element(self):
        blob = serialize_f32([42.0])
        result = deserialize_f32(blob)
        assert len(result) == 1
        assert abs(result[0] - 42.0) < 1e-5


class TestEmbedBatch:
    @pytest.mark.parametrize("is_query", [False, True])
    def test_calls_openai_correctly(self, is_query):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        mock_client.embeddings.create.return_value = mock_resp

        cfg = Config(
            embed_model="test-model",
            embed_dims=2,
            local_embed_query_prefix="query: ",
            local_embed_document_prefix="passage: ",
        )
        result = embed_batch(mock_client, ["text1", "text2"], cfg, is_query=is_query)

        mock_client.embeddings.create.assert_called_once_with(
            model="test-model", input=["text1", "text2"], dimensions=2
        )
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.parametrize(
        "is_query, prefix", [(True, "query: "), (False, "passage: ")]
    )
    def test_local_dispatch_applies_prefix(self, mock_local_model, is_query, prefix):
        cfg = Config(
            embed_method="local",
            local_embed_query_prefix="query: ",
            local_embed_document_prefix="passage: ",
        )

        result = embed_batch(None, ["hello"], cfg, is_query=is_query)

        mock_local_model.encode.assert_called_once_with(
            [prefix + "hello"], normalize_embeddings=True, prompt=""
        )
        assert result == [[0.1, 0.2, 0.3]]


class TestLocalEmbedBatch:
    @pytest.mark.parametrize("is_query", [False, True])
    def test_empty_prefixes_keep_plain_encoding(self, mock_local_model, is_query):
        result = local_embed_batch(["hello"], Config(), is_query=is_query)

        mock_local_model.encode.assert_called_once_with(
            ["hello"], normalize_embeddings=True
        )
        assert result == [[0.1, 0.2, 0.3]]

    def test_query_prefix(self, mock_local_model):
        cfg = Config(local_embed_query_prefix="query: ")

        local_embed_batch(["hello"], cfg, is_query=True)

        mock_local_model.encode.assert_called_once_with(
            ["query: hello"], normalize_embeddings=True, prompt=""
        )

    def test_document_prefix(self, mock_local_model):
        cfg = Config(local_embed_document_prefix="passage: ")

        local_embed_batch(["hello"], cfg)

        mock_local_model.encode.assert_called_once_with(
            ["passage: hello"], normalize_embeddings=True, prompt=""
        )

    @pytest.mark.parametrize(
        "is_query, prefix", [(True, "query: "), (False, "passage: ")]
    )
    def test_prefix_applies_to_entire_batch(self, mock_local_model, is_query, prefix):
        cfg = Config(
            local_embed_query_prefix="query: ",
            local_embed_document_prefix="passage: ",
        )
        texts = ["foo", "bar"]

        result = local_embed_batch(texts, cfg, is_query=is_query)

        mock_local_model.encode.assert_called_once_with(
            [prefix + "foo", prefix + "bar"], normalize_embeddings=True, prompt=""
        )
        assert texts == ["foo", "bar"]
        assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

    @pytest.mark.parametrize(
        "is_query, prefix", [(True, "query: "), (False, "passage: ")]
    )
    def test_explicit_prefix_overrides_model_prompts(
        self, mock_local_model, is_query, prefix
    ):
        mock_local_model.prompts = {
            "query": "built-in query: ",
            "document": "built-in document: ",
        }
        mock_local_model.default_prompt_name = "document"
        cfg = Config(
            local_embed_query_prefix="query: ",
            local_embed_document_prefix="passage: ",
        )

        local_embed_batch(["hello"], cfg, is_query=is_query)

        mock_local_model.encode.assert_called_once_with(
            [prefix + "hello"], normalize_embeddings=True, prompt=""
        )
        assert mock_local_model.default_prompt_name == "document"

    def test_empty_query_prefix_keeps_builtin_query_prompt(self, mock_local_model):
        mock_local_model.prompts = {"query": "built-in query: "}
        cfg = Config(local_embed_document_prefix="passage: ")

        local_embed_batch(["hello"], cfg, is_query=True)

        mock_local_model.encode.assert_called_once_with(
            ["hello"], normalize_embeddings=True, prompt_name="query"
        )

    def test_empty_document_prefix_keeps_default_prompt(self, mock_local_model):
        mock_local_model.prompts = {"query": "query: ", "document": "passage: "}
        mock_local_model.default_prompt_name = "document"
        cfg = Config(local_embed_query_prefix="query: ")

        local_embed_batch(["hello"], cfg)

        mock_local_model.encode.assert_called_once_with(
            ["hello"], normalize_embeddings=True
        )

    @pytest.mark.parametrize(
        "prefix", ["search_query: ", "Represent this query for retrieval: ", "検索: \n"]
    )
    def test_custom_prefix_is_used_verbatim(self, mock_local_model, prefix):
        cfg = Config(local_embed_query_prefix=prefix)

        local_embed_batch(["hello"], cfg, is_query=True)

        mock_local_model.encode.assert_called_once_with(
            [prefix + "hello"], normalize_embeddings=True, prompt=""
        )

    def test_prefix_keeps_dimension_truncation(self, mock_local_model):
        cfg = Config(embed_dims=2, local_embed_document_prefix="passage: ")

        result = local_embed_batch(["hello"], cfg)

        assert result == [[0.1, 0.2]]
