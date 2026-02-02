import matplotlib.pyplot as plt
import seaborn as sns
import torch

def visualize_attention(input_sentence, output_sentence, attention_weights, layer=None, head=None):
    """
    input_sentence: list of input tokens
    output_sentence: list of output tokens
    attention_weights: Tensor of shape [layers, heads, tgt_len, src_len]
    layer: which layer to visualize
    head: which head to visualize
    """
    if layer is None: layer = -1
    if head is None: head = 0

    attention = attention_weights[layer][head].detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, xticklabels=input_sentence, yticklabels=output_sentence, cmap='viridis')
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title(f"Attention Head {head}, Layer {layer}")
    plt.show()



# Предположим, вы уже декодировали 3 примера и получили весы внимания

for i in range(3):
    input_tokens = input_tokenizer.convert_ids_to_tokens(input_ids[i])
    output_tokens = output_tokenizer.convert_ids_to_tokens(output_ids[i])

    visualize_attention(
        input_sentence=input_tokens,
        output_sentence=output_tokens,
        attention_weights=attention_weights[i],  # если батч внимания
        layer=-1,  # последний слой
        head=0     # первая голова
    )