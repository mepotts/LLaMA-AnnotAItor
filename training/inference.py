import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from model_config import CONFIG
import json
from typing import Dict, List, Optional

class LyricsAnnotator:
    def __init__(
        self,
        base_model_path: str = CONFIG["model"].base_model_name,
        lora_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.config = CONFIG["inference"]
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True
        )
        
        # Load LoRA if provided
        if lora_path:
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                device_map="auto"
            )
    
    def generate_annotation(
        self, 
        lyrics: str,
        fragment: Optional[str] = None
    ) -> Dict:
        """
        Generate annotation for lyrics or specific fragment
        """
        # Prepare prompt
        if fragment:
            prompt = f"""
Analyze the following song lyric fragment:
Fragment: "{fragment}"
Context Lyrics: "{lyrics[:500]}"

Provide an insightful annotation that explains:
- Literary devices used
- Cultural or historical references
- Deeper meaning or interpretation
"""
        else:
            prompt = f"""
Analyze these song lyrics and identify interesting fragments:
Lyrics: "{lyrics}"

For each interesting fragment, provide:
- The fragment itself
- Literary devices used
- Cultural or historical references
- Deeper meaning or interpretation
"""
        
        # Generate
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=True
        )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse and structure the output
        response = self._parse_annotation(generated_text, fragment is not None)
        
        return {
            "input": {
                "lyrics": lyrics,
                "fragment": fragment
            },
            "annotations": response
        }
    
    def _parse_annotation(self, text: str, is_fragment: bool) -> List[Dict]:
        """Parse generated text into structured annotations"""
        annotations = []
        
        if is_fragment:
            # Single fragment annotation
            annotation = {
                "fragment": text.split('Fragment: "')[1].split('"')[0] if 'Fragment: "' in text else "",
                "analysis": text.split('interpretation\n')[-1].strip() if 'interpretation\n' in text else text
            }
            annotations.append(annotation)
        else:
            # Multiple fragment annotations
            fragments = text.split('Fragment: "')[1:]  # Skip the first split which is the prompt
            for fragment in fragments:
                try:
                    fragment_text = fragment.split('"')[0]
                    analysis = fragment.split('"')[-1].strip()
                    
                    annotation = {
                        "fragment": fragment_text,
                        "analysis": analysis
                    }
                    annotations.append(annotation)
                except:
                    continue
        
        return annotations

def main():
    # Example usage
    annotator = LyricsAnnotator(
        lora_path="../models/lora_adapters/final"
    )
    
    sample_lyrics = """
    In my life, I've seen miracles
    Each day a new surprise
    But nothing has touched my heart quite like
    The look in your eyes
    """
    
    # Generate annotations for full lyrics
    full_analysis = annotator.generate_annotation(sample_lyrics)
    print("\nFull Lyrics Analysis:")
    print(json.dumps(full_analysis, indent=2))
    
    # Generate annotation for specific fragment
    fragment = "In my life, I've seen miracles"
    fragment_analysis = annotator.generate_annotation(sample_lyrics, fragment)
    print("\nFragment Analysis:")
    print(json.dumps(fragment_analysis, indent=2))

if __name__ == "__main__":
    main()