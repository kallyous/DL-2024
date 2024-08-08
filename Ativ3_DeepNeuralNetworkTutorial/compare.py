'''Avaliação dos modelos da Atividade 3
Roda os modelos da atividade, coleta seus valores de retorno e compara.
Lucas Carvalho Flores
'''


import subprocess


def run_script(script_name):
    result = subprocess.run(['python', script_name, '--evaluate', '2>', '/dev/null'], capture_output=True, text=True)
    return float(result.stdout.strip())


def main():

    scripts = ['Ativ31LinRegSingVar.py',
               'Ativ32LinRegMultiVar.py',
               'Ativ33DeepNeuroNetSingVar.py',
               'Ativ34DeepNeuroNetMultiVar.py']
    results = {}

    for script in scripts:
        loss = run_script(script)
        results[script] = loss

    print("Resultados:")
    for script, loss in results.items():
        print(f"{script}: {loss}")


if __name__ == "__main__":
    main()
