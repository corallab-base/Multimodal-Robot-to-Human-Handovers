import multiprocessing
from pathlib import Path
import time

import torch
import spacy
from spacy import displacy
from spacy.symbols import pobj, dobj, ADP, NOUN, PROPN, nsubj

from gaze_utils.record_audio import Recorder

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

def group_compound_nouns(doc):
    compounds = []
    for token in doc:
        # If the token is a noun or proper noun (you might adjust this according to your needs)
        if token.pos_ in ['NOUN', 'PROPN']:
            # Initialize compound noun with the current token's text
            compound_noun = token.text
            head = token.head  # The token's syntactic parent to check for compounds
            
            # Iterate through the token's children to check for 'compound' relations
            for child in token.children:
                # print('child and dep', child, child.dep_)
                if child.dep_ == 'compound' or child.dep_ == 'amod':
                    compound_noun = child.text + ' ' + compound_noun  # Prepend the compound part
                    
            # Check if the head of the current noun is also a noun; if so, it's likely part of a compound noun
            if head.pos_ in ['NOUN', 'PROPN'] and token.dep_ == 'compound':
                continue  # Skip adding to the list, as it will be captured by the head noun processing
            
            compounds.append(compound_noun)
    return compounds

def parse_sentence(sentence, debug=False):
    import json
    import requests 

    API_TOKEN = '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' # Don't worry - burner account
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

    data = {
        'inputs': {
            'question': 'What object?',
            'context': sentence
        }
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        answer = response.json()
        object_name = answer['answer'].strip()
    else:
        print("Failed to retrieve:", response.text)
        raise Exception()

    # Preprocessing to remove POS terms that disconect the dependency parsing of Spacy
    sentence = sentence.replace('on to ', '')
    sentence = sentence.replace('onto ', '')
    sentence = sentence.replace('would like to ', '')
    sentence = sentence.replace('like to ', '')
    sentence = sentence.replace('want to ', '')

    doc = nlp(sentence)

    if debug:
        svg = displacy.render(doc, style='dep')
        output_path = Path(f"./gaze_utils/{sentence.replace(' ', '_')}.svg") 
        output_path.open("w", encoding="utf-8").write(svg)

    nouns = group_compound_nouns(doc)

    # print('object', object_name)
    # print('nouns', nouns)

    for noun in nouns:
        if noun.find(object_name) != -1:
            nouns.remove(noun)

    if len(nouns) == 1:
        part = nouns[0].strip()
    elif len(nouns) == 0:
        part = None
    else:
        raise Exception('Too many nouns in sentence (!=1)', sentence)

    '''
    If there is a part, who should hold it? 
    '''
    giving_parts = []
    holding_parts = []
    target_holder = None

    if part is not None:

        # Iterate through the tokens in the parsed sentence
        for token in doc:
            if token.lemma_.lower() in ['hand', 'give', 'pass']:
                dobjs = [child for child in token.children 
                        if child.dep == dobj and child.text.lower() not in ('it', 'me', 'i')]
                giving_parts += dobjs

            if token.lemma_.lower() in ['hold', 'grab', 'grasp']:
                dobjs = [child for child in token.children 
                        if child.dep == dobj and child.text.lower() not in ('it', 'me', 'i')]
                
                # We have hold, grab, and grasp which are words that sound like commands to the robot!
                # So it's a holding_part right! Unless it's "I grasp" or "I hold"

                prefix_I = any(child.dep == nsubj and child.text.lower() in ('me', 'i') for child in token.children)
                
                if prefix_I:
                    giving_parts += dobjs
                else:
                    holding_parts += dobjs

        giving_parts = list(map(str, giving_parts))
        holding_parts = list(map(str, holding_parts))

        if part in holding_parts:
            target_holder = 'robot'
        elif part in giving_parts:
            target_holder = 'human'
        else:
            # See which has more
            if len(holding_parts) > len(giving_parts):
                target_holder = 'robot'
            elif len(holding_parts) < len(giving_parts):
                target_holder = 'human'
            else:
                # For tie, default to robot
                target_holder = 'robot' 

    return object_name, part, target_holder


def parse_audio0(file="gaze_utils/2 Mustard.wav"):
    command = \
    'import whisper;' + \
    'model = whisper.load_model("base", device="cpu");' + \
    f'result = model.transcribe(\"{file}\");' + \
    'print(result["text"])'

    fullcommand = f'/home/corallab/anaconda3/condabin/conda run -n whisper python -c \'{command}\' '

    print(fullcommand)

    import subprocess
    result = subprocess.run(fullcommand, 
                            shell=True, stdout=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise Exception("Speech transcription failed")
    
    res = (str(result.stdout).strip().strip('.'))

    return res

def parse_audio(filename):
    import json
    import requests 

    API_TOKEN = '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' # Don't worry - burner account
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"

    def query(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    data = query(filename)

    print('HF returns:', data)

    return data['text']

# print('Making a dummy HF request to warm up to API...')
# multiprocessing.Process(target=parse_audio, args=('demo_audio.mp3',))

sentences = [
    "Grasp the yellow banana from tip and hand it to me",
    "Grasp the red apple and hand it over",
    "Hand me the mustard bottle by grabbing the tip",
    "Grab the rim of the wine glass and hand it to me",
    "Give me the rectangular box by the opening",
    "Hand me the tin can",
    "Give me the handle of the wooden hammer",
    "Give me the knife and hold the blade",
    # Failures
    "Give me the knife and hold on to the blade",
    "Grasp the drill and hand it over but I would like to grasp the handle",
]

'''
General syntax:

[Hand/Give/Pass/Hold/Grab/Pass] me the OBJECT [from/using/by the] PART

[Grasp/Give/Hand] me the OBJECT
'''

# for sentence in sentences:
#     print(f"{sentence}")
#     object, part, target_holder = parse_sentence(sentence)

#     if part is None:
#         print(f"    object: {object} \n    part: {part}")
#     else:
#         print(f"    object: {object} \n    part: {part} (held by {target_holder})")
#     print()

def record_and_parse(text=None, recording_done_func=None):
    '''
    Say something in the mic then Ctrl C

    pass text argument to override text transcription (fpr testing)
    
    returns object, part, target_holder
    '''
    if text is None:
        Recorder().record()

        recording_done_func()

        time.sleep(0.3)

        torch.cuda.empty_cache()
        text = parse_audio0('gaze_utils/audio.wav')
    else:
        print('Simulating prompt for 2 seconds')
        time.sleep(2)
        recording_done_func()
        
    print('Input Sentence:', text)
    object, part, target_holder = parse_sentence(text)

    if part is None:
        print(f"    object: {object} \n    part: {part}")
    else:
        part = str(part)
        if object.find(part) != -1:
            part = None
            print(f"    object: {object} \n    part: {part}")
        else:
            print(f"    object: {object} \n    part: {part} (held by {target_holder})")
            

    return object, part, target_holder

if __name__ == "__main__":
    # record_and_parse()
    res = parse_sentence("give me the red cup and I want to grasp the handle", debug=True)
    print(res)