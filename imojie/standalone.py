from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import import_submodules, JsonDict, sanitize
import_submodules('imojie')

def process(token_ids):
    temp=" ".join(token_ids)
    temp = temp.replace(" ##","")
    temp = temp.replace("[unused1]","( ")
    temp = temp.replace("[unused2]"," ; ")
    temp = temp.replace("[unused3]","")
    temp = temp.replace("[unused4]"," ; ")
    temp = temp.replace("[unused5]","")
    temp = temp.replace("[unused6]"," )")
    temp = temp.strip()
    temp = temp.split("[SEP]")
    ans=[]
    for x in temp:
        if x!="":
            ans.append(x)
    return ans

archive = load_archive(
    "models/imojie",
    weights_file="models/imojie/model_state_epoch_7.th",
    cuda_device=-1)

predictor = Predictor.from_archive(archive, "noie_seq2seq")
inp_sent = 'I ate an apple and an orange'
inp_instance = predictor._dataset_reader.text_to_instance(inp_sent)
output = predictor._model.forward_on_instance(inp_instance)
output = sanitize(output)
output = process(output["predicted_tokens"][0])
print(output)


