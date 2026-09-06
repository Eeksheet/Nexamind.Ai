from src.routing.question_router import Route, route_question
from src.web.search import _DDGParser


def test_router_general_current_rag_hybrid():
    assert route_question("What is photosynthesis?").route == Route.LLM_ONLY
    assert route_question("Who is the current Prime Minister of India?").route == Route.WEB_SEARCH
    assert route_question("According to the available documents, what happened to X?").route == Route.RAG
    assert route_question("Compare the knowledge base with the latest information online.").route == Route.HYBRID


def test_router_respects_disabled_capabilities():
    assert route_question("What is today's news?", web_enabled=False, rag_enabled=False).route == Route.LLM_ONLY
    assert route_question("According to the documents, what happened?", web_enabled=True, rag_enabled=False).route == Route.LLM_ONLY


def test_duckduckgo_parser_extracts_sources():
    html = '''<a class="result__a" href="https://example.com/a">Example title</a>
    <a class="result__snippet">A useful snippet about the result.</a>
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb">Second title</a>
    <a class="result__snippet">Second snippet.</a>'''
    p = _DDGParser(); p.feed(html); p.close()
    assert len(p.results) == 2
    assert p.results[0].title == "Example title"
    assert p.results[0].snippet.startswith("A useful snippet")
    assert p.results[1].url == "https://example.org/b"


def test_gemini_provider_import_and_name():
    from src.llm.providers import GeminiProvider
    assert GeminiProvider.name == "gemini"
