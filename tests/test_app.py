import sys
import types
import unittest
from pathlib import Path


if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = types.ModuleType("streamlit")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import src.app as app  # noqa: E402 # <- Comentário para o verificador de estilo flake8. #noqa = "no quality assurance" #E402 = regra que reclama quando um import não fica no topo do arquivo


class AppTests(unittest.TestCase):
    def setUp(self):
        self._original_values = {
            "_module_ingest_and_anonymize": app._module_ingest_and_anonymize,
            "_module_chunk_document": app._module_chunk_document,
            "_module_generate_embeddings": app._module_generate_embeddings,
            "_module_store_in_vector_db": app._module_store_in_vector_db,
            "_module_retrieve_context": app._module_retrieve_context,
            "_module_retrieve_relevant_chunks": app._module_retrieve_relevant_chunks,
            "_module_generate_rag_response": app._module_generate_rag_response,
            "_module_generate_response": app._module_generate_response,
            "DEFAULT_CHUNK_SIZE": app.DEFAULT_CHUNK_SIZE,
            "DEFAULT_CHUNK_OVERLAP": app.DEFAULT_CHUNK_OVERLAP,
            "st": app.st,
        }

        app._module_ingest_and_anonymize = None
        app._module_chunk_document = None
        app._module_generate_embeddings = None
        app._module_store_in_vector_db = None
        app._module_retrieve_context = None
        app._module_retrieve_relevant_chunks = None
        app._module_generate_rag_response = None
        app._module_generate_response = None
        app.DEFAULT_CHUNK_SIZE = 5
        app.DEFAULT_CHUNK_OVERLAP = 1
        app.st = types.SimpleNamespace(session_state={})

    def tearDown(self):
        for name, value in self._original_values.items():
            setattr(app, name, value)

# Validando a anonimização de dados sensíveis
    def test_ingest_and_anonymize_mascaras_dados_sensiveis(self):
        texto = (
            "Maria Silva CPF 123.456.789-00, email maria@exemplo.com, "
            "telefone (11) 98765-4321, cartao 4532 1122 3344 5566, senha: 123456"
        )

        resultado = app.ingest_and_anonymize(texto)

        self.assertIn("[CPF REMOVIDO]", resultado)
        self.assertIn("[EMAIL REMOVIDO]", resultado)
        self.assertIn("[TELEFONE REMOVIDO]", resultado)
        self.assertIn("[DADO BANCARIO REMOVIDO]", resultado)
        self.assertIn("[DADO REMOVIDO]", resultado)
        self.assertNotIn("123.456.789-00", resultado)
        self.assertNotIn("maria@exemplo.com", resultado)
        self.assertNotIn("(11) 98765-4321", resultado)

# Conferindo o chunking e os metadados.
    def test_chunk_document_quebra_texto_e_preserva_metadados(self):
        metadata = {"titulo": "Guia", "fonte": "base_interna"}

        chunks = app.chunk_document("abcdefghij", metadata)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["texto"], "abcde")
        self.assertEqual(chunks[1]["texto"], "efghi")
        self.assertEqual(chunks[2]["texto"], "ij")
        self.assertEqual(chunks[0]["metadata"], metadata)
        self.assertEqual(chunks[0]["doc_id"], chunks[1]["doc_id"])
        self.assertTrue(chunks[0]["id"].startswith(chunks[0]["doc_id"]))

# Verificando a geração simples de embeddings...
    def test_generate_embeddings_adiciona_vetor_simples(self):
        chunks = [{"id": "c1", "texto": "ola mundo ola"}]

        resultado = app.generate_embeddings(chunks)

        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado[0]["embedding"], {"ola": 2.0, "mundo": 1.0})
        self.assertEqual(resultado[0]["id"], "c1")

# Validando a persistência no estado local.
    def test_store_in_vector_db_salva_no_session_state(self):
        embedded_chunks = [{"id": "c1", "embedding": {"ola": 1.0}}]

        sucesso = app.store_in_vector_db(embedded_chunks)

        self.assertTrue(sucesso)
        self.assertEqual(app.st.session_state["vector_db"], embedded_chunks)
        self.assertTrue(app.st.session_state["vector_db_ready"])

# Validando a ordenação por similaridade.
    def test_retrieve_context_ordena_por_similaridade(self):
        app.st.session_state["vector_db"] = [
            {"id": "c1", "texto": "gato cachorro", "embedding": {"gato": 1.0, "cachorro": 1.0}},
            {"id": "c2", "texto": "banana laranja", "embedding": {"banana": 1.0, "laranja": 1.0}},
        ]

        resultado = app.retrieve_context("gato", top_k=2)

        self.assertEqual(resultado[0]["id"], "c1")
        self.assertGreater(resultado[0]["score"], resultado[1]["score"])

# Simulando triagem quando a evidência é fraca. Ou seja, quando o score é baixo, o sistema deve indicar que precisou fazer uma triagem e mencionar isso na resposta gerada.
    def test_generate_rag_response_aciona_triagem_quando_score_baixo(self):
        contexto = [{"id": "c1", "texto": "trecho", "metadata": {"titulo": "Doc A"}, "score": 0.1}]

        resultado = app.generate_rag_response("pergunta", contexto)

        self.assertTrue(resultado["precisou_triagem"])
        self.assertIn("triagem", resultado["resposta_gerada"].lower())

# Verificando a resposta com fontes citadas. Quando o score é alto, o sistema deve citar as fontes relevantes na resposta gerada.
    def test_generate_rag_response_cita_fontes_quando_score_ok(self):
        contexto = [{"id": "c1", "texto": "trecho relevante", "metadata": {"titulo": "Doc A"}, "score": 0.9}]

        resultado = app.generate_rag_response("pergunta", contexto)

        self.assertFalse(resultado["precisou_triagem"])
        self.assertIn("Doc A", resultado["resposta_gerada"])

# Coferindo a orquestração do pipeline RAG.
    def test_run_rag_pipeline_usa_retrieval_e_llm(self):
        original_retrieve = app.retrieve_context
        original_generate = app.generate_rag_response

        try:
            app.retrieve_context = lambda query, top_k=3: [
                {"id": "c1", "texto": "trecho", "metadata": {"titulo": "Doc A"}, "score": 0.8}
            ]
            app.generate_rag_response = lambda query, retrieved_context: {
                "resposta_gerada": "resposta final",
                "fontes": retrieved_context,
                "precisou_triagem": False,
                "confianca": 0.9,
            }

            resultado = app._run_rag_pipeline("pergunta")

            self.assertEqual(resultado["resposta_gerada"], "resposta final")
            self.assertEqual(len(resultado["fontes"]), 1)
            self.assertFalse(resultado["precisou_triagem"])
            self.assertEqual(resultado["confianca"], 0.9)
        finally:
            app.retrieve_context = original_retrieve
            app.generate_rag_response = original_generate


if __name__ == "__main__":
    unittest.main()
