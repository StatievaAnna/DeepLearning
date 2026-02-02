import torch

from preprocessing_emb import get_model_and_tokenizer, get_embeddings_and_masks

DEVICE = torch.device('cuda')

class Summarizer:
    """
    Генерирует заголовок по одному тексту, используя обученный EncoderDecoder и word_field.
    """
    def __init__(self, model, 
                 device=DEVICE,
                 max_len: int = 5000):
        tokenizer, model_emb = get_model_and_tokenizer() 
        self.model_emb = model_emb
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

        self.pad_idx = tokenizer.pad_token_id
        self.cls_idx = tokenizer.cls_token_id
        self.sep_idx = tokenizer.sep_token_id

        self.model.eval()

    def _encode(self, text: str) -> list[int]:

        encoded = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded['input_ids'].to(DEVICE)
        attention_mask = encoded['attention_mask'].to(DEVICE)

        with torch.no_grad():
            outputs = self.model_emb(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
        return embeddings, attention_mask

    def _decode(self, ids: list[int]) -> str:
        text = self.tokenizer.decode(ids, skip_special_tokens=False)

        # return ' '.join(tokens)
        return text

    def predict(self, text: str, return_attn=False) -> str:
        """
        text : сырой текст статьи/новости
        return: сгенерированная сводка
        """

        generated = [self.cls_idx]

        source_embeddings, source_mask = get_embeddings_and_masks(text)

        for _ in range(self.max_len):
            
            generated_text = self.tokenizer.decode(generated)
            target_embeddings, target_mask = get_embeddings_and_masks(generated_text)

            # tgt_tensor = torch.tensor(generated,
            #                            dtype=torch.long,
            #                            device='cpu').unsqueeze(0)
            source_embeddings = source_embeddings.to('cpu')
            target_embeddings = target_embeddings.to('cpu')
            source_mask = source_mask.to('cpu')
            target_mask = target_mask.to('cpu')

            with torch.no_grad():
                if return_attn == False:
                    logits = self.model(source_embeddings, target_embeddings, source_mask, target_mask)
                else:
                    logits, attn = self.model(source_embeddings, target_embeddings,  source_mask, target_mask, True)

                next_token = logits[0, -1].argmax(-1).item()
                # print(next_token)

                # if next_token == self.tokenizer.sep_token:
                #     break

            generated.append(next_token)
            if next_token == self.tokenizer.sep_token or len(generated) > 20:
                    break

        # Декодируем обратно в строку (без спецтокенов)
        if return_attn == False:
            return self._decode(generated[1:-1])
        else:
            return self._decode(generated[1:-1]), attn