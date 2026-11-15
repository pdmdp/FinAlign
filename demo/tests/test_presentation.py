import unittest
from html.parser import HTMLParser

from finalign.presentation import board_html, highlight, query_terms, summary_html


class ParsedHTML(HTMLParser):
    def __init__(self, markup):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.text = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, text):
        self.text.append(text)


class PresentationTests(unittest.TestCase):
    def test_highlighting_preserves_text_without_activating_markup(self):
        text = '<script>alert("bonds")</script> & <img src=x onerror="alert(1)">'
        parsed = ParsedHTML(highlight(text, query_terms("script bonds img")))
        self.assertEqual("".join(parsed.text), text)
        self.assertTrue(parsed.tags)
        self.assertEqual({tag for tag, _ in parsed.tags}, {"mark"})

    def test_custom_question_has_no_relevance_metrics_and_is_escaped(self):
        question = '<img src=x onerror="alert(1)">'
        parsed = ParsedHTML(summary_html(question, lexical=[("a", 8.0)], neural=[("b", 0.5)]))
        self.assertNotIn("img", [tag for tag, _ in parsed.tags])
        self.assertIn(question, "".join(parsed.text))
        self.assertIn("No judgments", "".join(parsed.text))
        self.assertNotIn("nDCG@10", "".join(parsed.text))

    def test_failed_neural_search_keeps_lexical_documents(self):
        class Index:
            documents = {"a": "First document", "b": "Second document"}

            def snippet(self, docno, passage=None):
                return self.documents[docno]

        parsed = ParsedHTML(board_html(
            Index(), "document", [("a", 8.0), ("b", 4.0)], state="failed",
            notice=("quota", "GPU quota reached", "Try again later."),
        ))
        articles = [attrs for tag, attrs in parsed.tags if tag == "article"]
        self.assertEqual(len(articles), 2)
        self.assertTrue(all("fa-m-bm25" in item["class"] for item in articles))
        self.assertIn("First document", "".join(parsed.text))
        self.assertIn("Second document", "".join(parsed.text))
        self.assertIn("GPU quota reached", "".join(parsed.text))


if __name__ == "__main__":
    unittest.main()
