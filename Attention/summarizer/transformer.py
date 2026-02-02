import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import wandb

from encoder_decoder import Encoder, Decoder
from preprocessing import convert_batch
from preprocessing_emb import convert_batch_with_pretrain_emb
from metrics import evaluate_rouge, log_rouge_to_wandb

DEVICE = torch.device('cuda')

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):

    def __init__(self, word_field, d_model=256, d_ff=1024,
                 blocks_count=4, heads_count=8, dropout_rate=0.1, embedding=None, tokenizer=None):
        super(EncoderDecoder, self).__init__()

        self.embedding=embedding
        self.d_model = d_model
        self.tokenizer=tokenizer
        self.word_field = word_field

        if self.word_field is not None:
            if word_field.vocab.vectors is not None:
                self.d_model = word_field.vocab.vectors.size(1)
                if self.d_model == 300:
                    heads_count = 10
            else:
                self.d_model = d_model
            self.word_field = word_field
            vocab_size = len(word_field.vocab)

            if embedding is not None:
                self.embedding = embedding
            else:
                self.embedding = nn.Embedding(vocab_size, self.d_model)
        
        if self.tokenizer is not None:
            vocab_size = len(self.tokenizer)
        else:
            vocab_size = None

        # print(f'self.d_model = {self.d_model}')

        self.encoder = Encoder(self.embedding, self.d_model, d_ff, blocks_count, heads_count, dropout_rate)
        self.decoder = Decoder(self.embedding, self.d_model, d_ff, blocks_count, heads_count, dropout_rate, vocab_size=vocab_size)
        # self.generator = Generator(d_model, target_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source_inputs, target_inputs, source_mask, target_mask, return_attn=False):
        encoder_output = self.encoder(source_inputs, source_mask, return_attn)
        decoder_output = self.decoder(target_inputs, encoder_output, source_mask, target_mask, return_attn)
        if return_attn:
            # Соберем все attention-слои
            attn_weights = {
                'encoder': self.encoder.self_attns,
                'decoder_self': [attn_weights[0] for attn_weights in self.decoder.self_attns],
                'decoder_enc': [attn_weights[1] for attn_weights in self.decoder.self_attns],
            }
            return decoder_output, attn_weights
        return decoder_output
    




def do_epoch(model, criterion, data_iter, optimizer=None, name=None, global_step=0):
    epoch_loss = 0
    is_train = optimizer is not None
    name = name or ''
    model.train(is_train)

    batches_count = len(data_iter)
    global_step = 0

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for i, batch in enumerate(data_iter):
                if model.embedding is not None:
                    source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch)
                else:
                    source_inputs, target_inputs, source_mask, target_mask = convert_batch_with_pretrain_emb(batch)

                logits = model.forward(
                    source_inputs, 
                    target_inputs[:, :-1], 
                    source_mask, 
                    target_mask[:, :-1, :-1]
                )

                logits_for_decoding = logits  # для декодирования до view

                logits = logits.contiguous().view(-1, logits.shape[-1])

                if model.tokenizer is not None:
                    encoded = model.tokenizer(
                        batch.target, 
                        padding=True, 
                        truncation=True, 
                        return_tensors='pt'
                    ).to(DEVICE)
                    target = encoded['input_ids']

                target = target[:, 1:].contiguous().view(-1)

                loss = criterion(logits, target)
                logits_ids = torch.argmax(logits_for_decoding, dim=-1)

                word_field = model.word_field

                if word_field is not None:
                    predicted_words = [word_field.vocab.itos[i] for i in logits_ids[0].detach().cpu().tolist()]
                    target_words = [word_field.vocab.itos[i] for i in target.view(-1)[:logits_ids.size(1)].detach().cpu().tolist()]

                    predicted_text = ' '.join(predicted_words)
                    target_text = ' '.join(target_words)
                else:
                    predicted_text = model.tokenizer.decode(
                        logits_ids[0].detach().cpu(),
                        skip_special_tokens=False  # как в оригинале
                    )
                    target_text = model.tokenizer.decode(
                        target.view(-1)[:logits_ids.size(1)].detach().cpu(),
                        skip_special_tokens=False  # как в оригинале
                    )

                rouge_scores = evaluate_rouge(predicted_text, target_text)

                epoch_loss += loss.item()

                if i % 10 == 0:
                    wandb.log({f'{name}_batch_loss': loss.item(), "step": global_step})
                    for k, v in rouge_scores.items():
                        wandb.log({f'{name}_batch_{k}': v, "step": global_step})
                    log_rouge_to_wandb(rouge_scores)

                global_step += 1

                if optimizer:
                    optimizer.optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                progress_bar.update()
                progress_bar.set_description(
                    '{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                        name, loss.item(), math.exp(loss.item()))
                )

            progress_bar.set_description(
                '{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                    name, epoch_loss / batches_count, math.exp(epoch_loss / batches_count))
            )
            progress_bar.refresh()

    avg_loss = epoch_loss / batches_count
    wandb.log({f'{name}_epoch_loss': avg_loss})
    return avg_loss, global_step
     


def fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None):

    global_step=0
    for epoch in range(epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        train_loss, global_step = do_epoch(model, criterion, train_iter, optimizer, name_prefix + 'Train:', global_step=global_step)
        if not val_iter is None:
            val_loss, _ = do_epoch(model, criterion, val_iter, None, name_prefix + '  Val:', global_step=global_step)

        wandb.log({
            'epoch':        epoch + 1,
            'train_loss':   train_loss,
            'val_loss':     val_loss
        }, step=global_step)

        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss
        }, step=epoch + 1)

    torch.save({
        'model_state_dict': model.state_dict(),
        'word_field': model.word_field
    }, 'model.pt')
    wandb.save('model.pt')

    wandb.finish()    