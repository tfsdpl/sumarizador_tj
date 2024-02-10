import bart_large_cnn
#import bart_large_cnn_fine_tune
import pegasus
import t5
import mt5
from datasets import list_datasets
from datasets import list_datasets, get_dataset_config_names

def datasets_br():
    available_datasets = list_datasets()
    portuguese_summarization_datasets = [ds for ds in available_datasets if
                                         "portuguese" in ds or "portugal" in ds or "brasil" in ds or "brazil" in ds]
    print(portuguese_summarization_datasets)


def dividir_texto_em_frases(texto):
    frases = texto.split(".")
    frases_limpa = [frase.strip() for frase in frases if frase.strip()]
    return frases_limpa


def datasets_br_sumarizacao():
    available_datasets = list_datasets()
    summarization_datasets = [ds for ds in available_datasets if "summarization" in ds]
    portuguese_summarization_datasets = {}
    for dataset in summarization_datasets:
        configs = get_dataset_config_names(dataset)
        for config in configs:
            if "portuguese" in config or "portugal" in config or "brasil" in config or "brazil" in config:
                portuguese_summarization_datasets[dataset] = config

    print(portuguese_summarization_datasets)




if __name__ == '__main__':
    texto = """DECIDO. Da análise dos autos, observo que o processo está regularmente instruído. Com efeito, tenho que, nestas circunstâncias, tornou-se desnecessário tecer maiores comentários sobre a temática abordada nesta demanda, já que, em virtude da nova redação dada ao §6º do artigo 226 da Constituição Federal, por meio da Emenda Constitucional nº 66/2010, não há que se falar em lapso temporal de separação judicial ou de fato do casal. Outrossim, constato que os direitos dos menores ficaram resguardados na avença firmada pelas partes, de maneira que não vejo óbice ao acolhimento da pretensão inicial. Isto posto, na confluência dessas considerações, HOMOLOGO o acordo firmado entre os autores, conforme discriminado no evento 12, para que surtam seus legais e jurídicos efeitos, decretando o divórcio de LUCIANA DA SILVA ABREU NASCIMENTO e ANTÔNIO CHAVES DO NASCIMENTO. Por conseguinte, JULGO EXTINTO O PROCESSO, com resolução de mérito, nos termos do artigo 487, III, b, do Código de Processo civil. DISPENSADO O PRAZO RECURSAL, NOS TERMOS DOS ART. 368 I A 368 L DA CAN (PROVIMENTO N. 002/2012 DA CGJ), DETERMINO A EXPEDIÇÃO DE DUAS VIAS DESSA SENTENÇA PARA SERVIR DE MANDADO DE AVERBAÇÃO DO DIVÓRCIO, FICANDO DESDE JÁ DETERMINADO AO OFICIAL DE REGISTRO CIVIL COMPETENTE QUE PROCEDA A NECESSÁRIA AVERBAÇÃO À MARGEM DO REGISTRO DE CASAMENTO DE MATRÍCULA Nº 024729 01 55 2013 2 00203 173 0041149 55 OBSERVANDO-SE OS DADOS DO PROCESSO E DAS PARTES ACIMA INDICADOS. A requerente voltará a usar o seu nome de solteira, qual seja, LUCIANA DA SILVA ABREU. Da mesma forma expeçam-se vias da sentença para servir de Termo de Guarda Definitiva à parte indicada no acordo homologado, ficando desde já o guardião devidamente compromissado ao exarar o recebimento, com a obrigação de prestar assistência material, moral e educacional ao(s) aludido(s) menor(es), mantendo-o(s) sob sua guarda e vigilância, devendo apresentá-lo(s) neste Juízo sempre que for exigida a sua presença, esclarecendo que a guarda ora conferida dá ao seu detentor o direito de opor-se a terceiros, inclusive aos pais (art. 33 do ECA), conferindo, ainda, a criança ou adolescente, a condição de dependente, para todos os fins de direito, inclusive previdenciários, de conformidade com o disposto no art. 33, § 3º do ECA. A sentença também servirá como Termo de Visitas, a ser exercida e cumprida conforme regulamentação contida no acordo homologado, ficando as partes advertidas que o descumprimento de qualquer delas importará em declaração de indícios de atos de alienação parental, nos termos do art. 4º da Lei n. 12.318/10, ficando a parte infratora sujeita às medidas previstas no art. 6º da mesma lei. Custas finais a serem suportadas pelas partes, ficando sua cobrança, no entanto, suspensa, por serem estes beneficiários da justiça gratuita (art. 98 do CPC). Sem honorários. Após certificado o trânsito em julgado da sentença, arquivem-se os autos com as devidas baixas, anotações e cautelas de estilo. Publique-se. Registre-se. Intimem-se. Aparecida de Goiânia, data e hora da assinatura eletrônica.   Mariuccia Benicio Soares Miguel Juíza de Direito 1  """
    # score = mt5.MT5Scorer()
    # score.sentence_score(texto)

    score = pegasus.PegasusScorer()
    score.sentence_score(texto)

    # bart_large_cnn.sentence_score(texto)