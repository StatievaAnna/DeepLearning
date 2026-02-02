import seaborn as sns
import matplotlib.pyplot as plt

def visualize_attention(input_sentence, output_sentence, attention_weights):

    attention = attention_weights.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, xticklabels=input_sentence, yticklabels=output_sentence, cmap='viridis')
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title(f"Attention map {type}")
    plt.show()

def attention_map(summarizer, input_text, type):
    predict, attn = summarizer.predict(input_text, True)

    print(f'predict: {predict}')

    attn = attn[type][-1][0]
    avg_attn = attn.mean(dim=0)
    tokenizer = summarizer.tokenizer

    if type == 'encoder':

        input_ids = tokenizer(input_text)['input_ids']
        input_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        input_tokens = input_tokens[:20]
        avg_attn = avg_attn[:20, :20] 

        visualize_attention(
        input_sentence=input_tokens,
        output_sentence=input_tokens,
        attention_weights=avg_attn)

    if type == 'decoder_self':

        output_ids = tokenizer(predict)['input_ids']
        output_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in output_ids]
        print(output_tokens)

        visualize_attention(
        input_sentence=output_tokens,
        output_sentence=output_tokens,
        attention_weights=avg_attn)


    if type == 'decoder_enc':

        input_ids = tokenizer(input_text)['input_ids']
        input_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        input_tokens = input_tokens[:20]
        avg_attn = avg_attn[:20, :20] 

        output_ids = tokenizer(predict)['input_ids']
        output_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in output_ids]

        visualize_attention(
            input_sentence=input_tokens,
            output_sentence=output_tokens,
            attention_weights=avg_attn, 
        )