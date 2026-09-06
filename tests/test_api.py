import pytest
from fastapi.testclient import TestClient

from src.api.app import Service, create_app


@pytest.fixture(scope="module")
def client(cfg):
    return TestClient(create_app(Service(cfg)))


def test_health_and_examples(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["provider"] == "mock"
    ex = client.get("/examples?n=3").json()
    assert len(ex) == 3 and {"qid", "question", "answer"} <= set(ex[0])


def test_query_with_dataset_context_is_grounded(client):
    ex = client.get("/examples?n=1").json()[0]
    r = client.post("/query", json={"question": ex["question"], "qid": ex["qid"],
                                    "config": {"orchestrator": {"budget": {"max_hops": 2}}}})
    assert r.status_code == 200, r.text
    j = r.json()
    assert {"answer", "confidence", "evidence", "trace", "metrics", "termination"} <= set(j)
    assert j["metrics"]["hops"] <= 2 and j["metrics"]["llm_calls"] >= 1
    assert j["backend"]["is_mock"] is True
    assert j["corpus"]["source"].startswith("dataset:")
    assert all({"title", "sent_idx", "sentence"} <= set(e) for e in j["evidence"])
    # cited supporting units must come from the retrieved evidence
    units = {(e["title"], e["sent_idx"]) for e in j["evidence"]}
    cited = {(u[0], u[1]) for t in j["trace"] if t["agent"] == "reasoner" for u in t.get("supporting_units", [])}
    assert cited <= units


def test_query_with_explicit_paragraphs_and_system_choice(client):
    body = {"question": "Where is Alpha?", "config": {"system": "single_pass"},
            "paragraphs": {"Alpha": ["Alpha is a river in Spain."], "Beta": ["Beta is a mountain."]}}
    j = client.post("/query", json=body).json()
    assert j["system"] == "single_pass" and j["corpus"]["source"] == "request" and j["corpus"]["n_paragraphs"] == 2


def test_query_validation_errors(client):
    assert client.post("/query", json={"question": "x"}).status_code == 422
    assert client.post("/query", json={"question": "who?", "config": {"system": "nope"}}).status_code == 400
    assert client.post("/query", json={"question": "who?", "config": {"retrieval": {"bad": 1}}}).status_code == 400
    assert client.post("/query", json={"question": "who?", "qid": "missing"}).status_code == 404


def test_index_html(client):
    r = client.get("/")
    assert r.status_code == 200 and "Nexamind.Ai" in r.text


def test_nexamind_chat_and_memory(client):
    r = client.post("/api/chat", json={"message": "What is photosynthesis?", "web_enabled": False, "rag_enabled": False})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["conversation_id"] and j["route"] == "LLM_ONLY"
    cid = j["conversation_id"]
    r2 = client.post("/api/chat", json={"message": "And why is it important?", "conversation_id": cid, "web_enabled": False, "rag_enabled": False})
    assert r2.status_code == 200
    hist = client.get(f"/api/chats/{cid}").json()
    assert len(hist["messages"]) == 4


def test_router_preview(client):
    r = client.get("/api/router", params={"question": "What are the latest AI developments?"})
    assert r.status_code == 200 and r.json()["route"] == "WEB_SEARCH"
