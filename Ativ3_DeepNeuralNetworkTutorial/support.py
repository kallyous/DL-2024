
import os
import requests


# Variável global para controle de debug
DEBUG = False

def log(*args, **kwargs):
    """
    Função de log que imprime mensagens somente se DEBUG estiver definido como True.
    :param args: Argumentos posicional que são passados para a função print.
    :param kwargs: Argumentos nomeados que são passados para a função print.
    """
    if DEBUG:
        print(*args, **kwargs)


def set_debug(dbg):
    DEBUG = dbg


def download(url, save_path):
    '''Recebe url de arquivo para baixar e salva em save_path.'''
    if not os.path.exists(save_path):
        log('Obtendo arquivo de', url)
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            log('Arquivo baixado e salvo em', save_path)
        else:
            raise Exception(f'ERRO: Problema ao baixar arquivo:\n{response.status_code}')
    else:
        log('Arquivo já está presente em', save_path)
