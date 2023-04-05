# logging
import logging;
import sys

logname = "log.txt";

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(logname),
        logging.StreamHandler()
    ])

# imports
import torch
from peft import PeftModel
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, AutoModelForTokenClassification
from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper, LLMPredictor, \
    ServiceContext
from typing import Optional, List, Mapping, Any
from peft import PeftModel

modelName = "decapoda-research/llama-7b-hf";
# lora_weights = "chansung/alpaca-lora-13b"; dos not work
# lora_weights = "baruga/alpaca-lora-13b";
# lora_weights = "mattreid/alpaca-lora-13b";
# lora_weights = "Dogge/alpaca-lora-13b";
# lora_weights = "circulus/alpaca-lora-13b";
# lora_weights = "daviddmc/lpaca-lora-13b";
# lora_weights = "mattreid/alpaca-lora-7b";
lora_weights = "tloen/alpaca-lora-7b";

# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
# Tokens
tokenizer = LlamaTokenizer.from_pretrained(modelName)

# ## Load base Model
model = LlamaForCausalLM.from_pretrained(
    modelName,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)
# #Load fine Tune
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float16,
)

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

# Compile does it need it?
#model = torch.compile(model)

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)


class CustomLLM(LLM):
    logging.info("cuda is available: " + str(torch.cuda.is_available()));
    # pipeline = pipeline(TASK,
    #                 model=MODEL_PATH,
    #                 device=1,     # to utilize GPU cuda:1
    #                 device=0,     # to utilize GPU cuda:0
    #                 device=-1)    # default value which utilize CPU
    # model = AutoModelForTokenClassification.from_pretrained(modelName)
    # pipeline = pipeline("text-generation", modelName,tokenizer = tokenizer, device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})

    pipeline = pipeline(
        "text-generation",
        device=-1,
        model=model,
        tokenizer=tokenizer,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.2
    )

    # huggingFacePipeline = HuggingFacePipeline(pipeline=pipeline)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = self.pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]

        # only return newly generated tokens
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"


logging.info("Debug log 1");
# define our LLM
llm_predictor = LLMPredictor(llm=CustomLLM())

logging.info("Debug log 2");
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

logging.info("Debug log 3");
# Load the your data
documents = SimpleDirectoryReader('./documentation').load_data()
index = GPTListIndex.from_documents(documents, service_context=service_context)

# Query and print response
response = index.query("What do you think of MITRE?",
                       service_context=service_context)

logging.info("response");
print(response)
