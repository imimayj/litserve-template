import os, time, json, unidecode, contractions, re
from typing import Any, List, Dict, Optional

import dotenv
import litserve as ls
import mlflow
from pydantic import BaseModel, Field

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

dotenv.load_dotenv()

class ClassificationResult(BaseModel):
    id: int
    label: str
    confidence: float
    source: str

class ClassifyRequest(BaseModel):
    id: int
    texts: List[str] = Field(default_factory=list)

class ClassifyResponse(BaseModel):
    id: int
    classifications: List[ClassificationResult]
    
class LitTCAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.mlflow_model, self.huggingface_model, self.tokenizer = self.init_model()
        self.model.to(self.device)
        self.class_labels = self.load_class_labels()
        self.top2 = os.getenv("RETURN_TOP2", "false").lower() == "true"
        
    def init_model(self):
        mlflow_path = os.getenv('MLFLOW_MODEL_PATH', "your-mlflow-uri")
        hf_model = os.getenv('HUGGINGFACE_MODEL', "your-huggingface-model")
        hf_tok = os.getenv('HUGGINGFACE_TOKENIZER', "your-model's-tokenizer")

        if mlflow_path:
            self.mlflow_model = mlflow.pytorch.load_model(mlflow_path, map_location='cpu')  
            self.tokenizer = AutoTokenizer.from_pretrained(hf_tok)
        
        elif hf_model:
            self.huggingface_model = AutoModelForSequenceClassification.from_pretrained(hf_model)
            self.tokenizer = AutoTokenizer.from_pretrained(hf_tok)
        else:
            raise ValueError("No model path provided. Set MLFLOW_MODEL_PATH or HUGGINGFACE_MODEL environment variable.")

    def load_class_labels(self):
        """
        Expects output_mapping.json in the server directory:
        [
          {"label": "AI"},
          {"label": "ML"},
          ...
        ]
        """
        mapping_path = os.path.join(os.path.dirname(__file__), "output_mapping.json")
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        return [row["label"] for row in mapping]

    def decode_request(self, request: ClassifyRequest) -> List[ClassifyRequest]:
        self.log("decode_request", {"payload": request.model_dump()})
        return [request]

    def predict(self, records: List[ClassifyRequest]):
        log_request_ids = os.getenv("LOG_REQUEST_IDS", "false").lower() == "true"
        start_time = time.time() if log_request_ids else None

        responses: List[Dict[str, Any]] = []

        for record in records:
            rec_id = record.id
            texts = [self.preprocess_text(t) for t in record.texts if isinstance(t, str) and t.strip()]

            if not texts:
                responses.append({"id": rec_id, "classifications": []})
                continue

            # Collect per-text top predictions; then pick the best text by summed confidence
            per_text_preds: List[List[Dict[str, Any]]] = []

            for text in texts:
                inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = getattr(outputs, "logits", outputs[0].logits if isinstance(outputs, tuple) else None)
                    probs = torch.softmax(logits, dim=1)  # [1, num_classes]
                    k = 2 if self.top2 else 1
                    topk_conf, topk_idx = torch.topk(probs, k=k, dim=1)

                preds = []
                for i in range(k):
                    label_idx = int(topk_idx[0][i].item())
                    conf = float(topk_conf[0][i].item())
                    value_name = self.class_labels[label_idx] if label_idx < len(self.class_labels) else "Unknown"
                    preds.append({"label": label_idx, "value_name": value_name, "confidence": conf, "source": "ml"})
                per_text_preds.append(preds)

            best = self._pick_best(per_text_preds)
            responses.append({"id": rec_id, "classifications": best})

            
            elapsed = time.perf_counter() - start_time
            token_count = sum(len(t.split()) for t in texts)
            resp_size = len(str(best).encode("utf-8"))
            self.log("model_metrics", {
                "model_name": getattr(self.model, "name_or_path", "ml-model"),
                "tokenizer": getattr(self.tokenizer, "name_or_path", "tokenizer"),
                "max_tokens": getattr(self.tokenizer, "model_max_length", None),
                "latency": elapsed,
                "tokens": token_count,
                "response_size": resp_size,
            })

            total = time.perf_counter() - start_time
            self.log("summary", f"Processed {len(records)} records in {total:.2f}s")

        return responses

    def encode_response(self, output) -> List[ClassifyResponse]:
        # Validate and shape with Pydantic
        return [ClassifyResponse(**r) for r in output]
    
    def preprocess_texts(self, text: str) -> str:
        
        self.log("original_text", text)
        text = text.lower()
        text = unidecode.unidecode(text)
        text = contractions.fix(text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z0-9:£$-,%.?! ]+", " ", text)
        text = re.sub(r'\s+', ' ', text).strip()

        self.log("processed_text", text)
        return text

    def _pick_best(self, all_preds: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Choose the text whose prediction list has the highest sum of confidences.
        """
        valid = [p for p in all_preds if p]
        if not valid:
            return []
        return max(valid, key=lambda lst: sum(d["confidence"] for d in lst))
