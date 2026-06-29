# Milestone 4 gate: the server speaks the OpenAI /v1/completions shape, streams over SSE,
# and greedy (temperature 0) output is deterministic and consistent between the two modes
import json

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture(scope="module")
def client():
    # the context manager runs the lifespan: loads the model, starts the step loop
    with TestClient(app) as c:
        yield c


def test_non_streaming_shape(client):
    r = client.post("/v1/completions",
                    json={"prompt": "2 + 2 =", "max_tokens": 8, "temperature": 0.0})
    assert r.status_code == 200
    body = r.json()

    assert body["object"] == "text_completion"
    choice = body["choices"][0]
    assert isinstance(choice["text"], str) and choice["text"]
    assert choice["index"] == 0
    assert choice["finish_reason"] in {"stop", "length"}
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_greedy_is_deterministic(client):
    payload = {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0.0}
    a = client.post("/v1/completions", json=payload).json()
    b = client.post("/v1/completions", json=payload).json()
    assert a["choices"][0]["text"] == b["choices"][0]["text"]


def test_streaming_matches_non_streaming(client):
    payload = {"prompt": "Once upon a time,", "max_tokens": 10, "temperature": 0.0}

    full = client.post("/v1/completions", json=payload).json()["choices"][0]["text"]

    payload["stream"] = True
    pieces, saw_done = [], False
    with client.stream("POST", "/v1/completions", json=payload) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                saw_done = True
                continue
            chunk = json.loads(data)
            assert chunk["object"] == "text_completion"
            pieces.append(chunk["choices"][0]["text"])

    assert saw_done
    assert "".join(pieces) == full      # the streamed deltas reassemble the full completion
