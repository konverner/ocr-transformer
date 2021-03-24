import torch
import random

class Hparams():
    def __init__(self):

        # SETS OF CHARACTERS
        self.cyrillic = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', 'EOS']
        # Символы, которые надо удалить
        self.del_sym = []

        self.lr = 1e-4
        self.batch_size = 8
        self.hidden = 512
        self.enc_layers = 1
        self.dec_layers = 1
        self.nhead = 4
        self.dropout = 0.1
        
        # IMAGE SIZE
        self.width = 256
        self.height = 64

class Paths():
    def __init__(self):
        # log folder
        self.log = r"/content/drive/MyDrive/log/"

        # checkpoint for training
        self.chk = None

        self.pretrain_labels_dir = r"/content/werner_pretrain/labels.tsv"
        self.pretrain_image_dir = r"/content/werner_pretrain/pretrain/"

        self.train_labels_dir = r"/content/real_cyr_dataset/real_cyr_data.tsv"
        self.train_image_dir = r"/content/real_cyr_dataset/real_cyr_data/"

        self.test_labels_dir = r'/content/OCR_DATASET_146k/OCR_DATASET_SYNTH_146k.txt'
        self.test_image_dir = r'/content/OCR_DATASET_146k/OCR_DATASET_SYNTH_146k/'


random.seed(1488)
torch.manual_seed(1488)
torch.cuda.manual_seed(1488)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hp = Hparams()
path = Paths()
