

class Config(object):
    # Paths
    caption_path = './data/annotations/eng/captions_train2017.json'
    validation_path = './data/annotations/eng/captions_val2017.json'
    vocab_path = './data/vocab.pkl'
    val_img_path = './data/val2017_resized/'
    train_img_path = './data/train2017_resized'
    machine_output_path = './data/july_27_v4_machine_output_eng_base.json'
    threshold = 1

    # hyperparams
    grad_clip = 5.
    num_epochs = 30
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
    encoder_path = "./checkpoints/encoder_20_july_22_eng_bert.ckpt"
    decoder_path = "./checkpoints/decoder_20_july_22_eng_bert.ckpt"