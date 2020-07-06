

class Config(object):
    # Paths
    caption_path = './data/annotations/captions_train2017.json'
    validation_path = './data/annotations/captions_val2017.json'
    vocab_path = './data/vocab.pkl'
    val_img_path = './data/val2017_resized/'
    machine_output_path = './data/july_v1_machine_output.json'
    threshold = 1

    # hyperparams
    grad_clip = 5.
    num_epochs = 20
    batch_size = 32
    decoder_lr = 0.0004

    # if both are false them model = baseline
    glove_model = False
    bert_model = True

    from_checkpoint = False
    train_model = True
    valid_model = False

    tokenizer = 'nltk'

    # vocab indices
    PAD = 0
    START = 1
    END = 2
    UNK = 3

    ##encoder params
    encoder_dim = 2048
    attention_dim = 512
    bert_embed_dim = 768
    glove_embed_dim = 300 
    embed_dim = 512
    decoder_dim = 512
    dropout = 0.5

    ##model path
    model_dir = "checkpoints/"
    encoder_path = "./checkpoints/encoder_epoch15-july-5"
    decoder_path = "./checkpoints/decoder_epoch15-july-5"