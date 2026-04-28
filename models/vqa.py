# ============================================================
# VQA Model: LLaVA, BLIP-2, BLIP
# ============================================================

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BlipProcessor,
    BlipForQuestionAnswering,
)
import warnings
warnings.filterwarnings("ignore")

class VQAModel:
    """
    Visual Question Answering using state-of-the-art models.
    Supports: LLaVA, BLIP-2, BLIP
    """
    
    def __init__(self, model_type="llava", device="cuda"):
        """
        Initialize VQA model.
        
        Args:
            model_type (str): "llava", "blip2", or "blip"
            device (str): "cuda" or "cpu"
        """
        self.device = device
        self.model_type = model_type
        
        print(f"Loading {model_type.upper()} VQA model...")
        
        if model_type == "llava":
            self._load_llava()
        elif model_type == "blip2":
            self._load_blip2()
        elif model_type == "blip":
            self._load_blip()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_llava(self):
        """
        Load LLaVA model (best quality, most reasoning).
        Requires: pip install git+https://github.com/haotian-liu/LLaVA.git
        """
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            
            model_path = "liuhaotian/llava-v1.5-7b"
            model_name = get_model_name_from_path(model_path)
            
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device_map="auto" if self.device == "cuda" else "cpu"
            )
            
            self.processor = None  # LLaVA uses image_processor instead
            print("✅ LLaVA loaded successfully")
        except ImportError:
            print("⚠️  LLaVA not installed. Install with:")
            print("   pip install git+https://github.com/haotian-liu/LLaVA.git")
            print("   Falling back to BLIP-2")
            self._load_blip2()
    
    def _load_blip2(self):
        """
        Load BLIP-2 model (good balance of quality and speed).
        """
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        model_name = "Salesforce/blip2-opt-6.7b"
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = None
        print("✅ BLIP-2 loaded successfully")
    
    def _load_blip(self):
        """
        Load original BLIP model (faster, good for real-time).
        """
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = None
        print("✅ BLIP loaded successfully")
    
    def answer_question(self, image_path, question, max_length=100):
        """
        Answer a question about an image.
        
        Args:
            image_path (str): Path to image
            question (str): Question in English or Vietnamese
            max_length (int): Max tokens in answer
            
        Returns:
            str: Answer text
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return f"Error loading image: {e}"
        
        if self.model_type == "llava":
            return self._answer_llava(image, question, max_length)
        elif self.model_type == "blip2":
            return self._answer_blip2(image, question, max_length)
        else:  # blip
            return self._answer_blip(image, question, max_length)
    
    def _answer_llava(self, image, question, max_length):
        """
        Generate answer using LLaVA.
        """
        try:
            from llava.conversation import conv_templates
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            
            # Prepare image
            image_tensor = self.image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            
            if self.device == "cuda":
                image_tensor = image_tensor.half().to(self.device)
            else:
                image_tensor = image_tensor.to(self.device)
            
            # Prepare conversation
            conv = conv_templates["v1"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0),
                    max_new_tokens=max_length,
                    use_cache=True,
                    temperature=0.7
                )
            
            answer = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            return answer
        except Exception as e:
            return f"Error in LLaVA inference: {e}"
    
    def _answer_blip2(self, image, question, max_length):
        """
        Generate answer using BLIP-2.
        """
        try:
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=3,
                    temperature=0.7
                )
            
            answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
            return answer
        except Exception as e:
            return f"Error in BLIP-2 inference: {e}"
    
    def _answer_blip(self, image, question, max_length):
        """
        Generate answer using BLIP.
        """
        try:
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=3
                )
            
            answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
            return answer
        except Exception as e:
            return f"Error in BLIP inference: {e}"
    
    def answer_multiple_questions(self, image_path, questions):
        """
        Answer multiple questions about the same image.
        
        Args:
            image_path (str): Path to image
            questions (list): List of questions
            
        Returns:
            dict: Question -> Answer mapping
        """
        results = {}
        for q in questions:
            results[q] = self.answer_question(image_path, q)
        return results
