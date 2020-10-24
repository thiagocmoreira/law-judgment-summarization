# !pip install transformers==3.1 --user
# !pip install pandas --user
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import logging
logging.basicConfig(level=logging.ERROR)

def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()
    
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# !pip install wandb
import wandb

project_name = 'pegasus-sum400-summarized'

device = 'cuda'

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.ementa = self.data.ementa
        self.inteiro_teor = self.data.inteiro_teor

    def __len__(self):
        return len(self.ementa)

    def __getitem__(self, index):
        inteiro_teor = str(self.inteiro_teor[index])
        inteiro_teor = ' '.join(inteiro_teor.split())

        ementa = str(self.ementa[index])
        ementa = ' '.join(ementa.split())

        source = self.tokenizer.batch_encode_plus([inteiro_teor], max_length=self.source_len, pad_to_max_length=True,return_tensors='pt', truncation=True)
        target = self.tokenizer.batch_encode_plus([ementa], max_length=self.summ_len, pad_to_max_length=True,return_tensors='pt', truncation=True)

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }
    
def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        loss = outputs[0]
        
        empty_cache()
        
        if _ % 10 == 0:
            wandb.log({"Loss do treinamento": loss.item()})

        if _ % 500 == 0:
            print(f'Época: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        empty_cache()
        
import os
# os.environ['WANDB_API_KEY'] = 'key'

def init_wandb(project_name):
    wandb.init(project=project_name) 
    config = wandb.config
    config.TRAIN_BATCH_SIZE = 1
    config.VALID_BATCH_SIZE = 1
    config.TRAIN_EPOCHS = 4
    config.VAL_EPOCHS = 1
    config.LEARNING_RATE = 1e-4
    config.MAX_LEN = 1024
    config.SUMMARY_LEN = 400

    return config

def init_training_pipeline(config):
    print('Iniciando pipeline...\n\n')

    model_name = 'google/pegasus-xsum'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    empty_cache()
    
    train_dataset = pd.read_csv('translated-train-400-2500-summarized-new.csv', encoding='utf-8', error_bad_lines=False, engine="python")
    train_dataset = train_dataset[['ementa','inteiro_teor']]
    print('Exemplo de textos:')
    print(train_dataset.head(), '\n\n')
    
    val_dataset = pd.read_csv('translated-validate-400-2500-summarized-new.csv', encoding='utf-8', error_bad_lines=False, engine="python")
    val_dataset = val_dataset[['ementa','inteiro_teor']]

    print(f'Dataset de treino: {train_dataset.shape}')
    print(f'Dataset de teste: {val_dataset.shape}')

    training_set = CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

    train_params = {'batch_size': config.TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    val_params = {'batch_size': config.VALID_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    
    print('Instanciando modelo...')
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

    empty_cache()

    wandb.watch(model, log="all")
    
    print('Inicializando Fine-Tuning utilizando o dataset de acórdãos...')

    for epoch in range(config.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

        empty_cache()

    print('Treinamento concluído!\n\n')

    return tokenizer, model, val_loader

config = init_wandb(project_name)

import time
start_time = time.time()

tokenizer, model, val_loader = init_training_pipeline(config)

print(f'--- Treino: {(time.time() - start_time)} seconds ---')

open("predictions.csv","w+")

def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    source_texts = []
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data['source_ids'].to(device, dtype=torch.long)
            y = data['target_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=250,
                num_beams=2,
                repetition_penalty=1.5,
                length_penalty=1,
                early_stopping=True
            )
            
            src_texts = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in ids]
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            if _ % 100 == 0 and _ > 0:
                print(f'Resumos: {_} gerados')
            
            source_texts.extend(src_texts)
            predictions.extend(preds)
            actuals.extend(target)

    return source_texts, predictions, actuals

def validade_and_save_predictions(val_epochs, tokenizer, model, val_loader):
    print('Gerando sumários utilizando o modelo no dataset de validação...')
    for epoch in range(val_epochs):
        source_texts, predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({ 'inteiro_teor_sumarizado': source_texts, 'ementa_original': actuals, 'resumo_gerado': predictions })
        final_df.to_csv('predictions.csv')
        print('CSV para análise gerado!')

val_start_time = time.time()

validade_and_save_predictions(config.VAL_EPOCHS, tokenizer, model, val_loader)
print(f'--- Validação: {(time.time() - val_start_time)} seconds ---')

generated_summaries = pd.read_csv('predictions.csv', encoding='utf-8', error_bad_lines=False, engine="python")
for index, row in generated_summaries[10:30].iterrows():
    print(f'Exemplo {index}', '\n')
    print('Inteiro teor:', row['inteiro_teor_sumarizado'], '\n\n')
    print('Ementa original:', row['ementa_original'], '\n\n')
    print('Sumário gerado:', row['resumo_gerado'], '\n\n')
    
print('Fim do script!', '\n\n')
