from pathlib import Path
import time

import torch
import spacy
from spacy import displacy
from spacy.symbols import pobj, dobj, ADP, NOUN, PROPN

from gaze_utils.record_audio import Recorder

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

def parse_sentence(sentence):
    doc = nlp(sentence)

    # svg = displacy.render(doc, style='dep')

    # output_path = Path(f"./gaze_utils/{sentence.replace(' ', '_')}.svg") 
    # output_path.open("w", encoding="utf-8").write(svg)
    
    '''
    Deduce the obj and part
    '''

    nouns = []
    preps = []
    for i, token in enumerate(doc):
        if token.pos in (NOUN, PROPN):
            nouns.append((i, token))
        elif token.pos == ADP:
            preps.append((i, token))    

    assert len(nouns) > 0, "Need at least one noun"

    nouns_first = nouns[0][0]
    nouns_last = nouns[-1][0]

    # find FIRST preposition between nouns
    for i, prep in preps:
        if nouns_first < i and i < nouns_last:
            preceding_noun = [noun for j, noun in nouns if j < i][-1]
            succeeding_noun = [noun for j, noun in nouns if i < j][0]
            the_actual_prep = prep

            # print('Got structure', preceding_noun, the_actual_prep, succeeding_noun)

            if the_actual_prep.text == 'of':
                object = succeeding_noun
                part = preceding_noun.text
            else:
                object = preceding_noun
                part = succeeding_noun.text

            break
    else:
        if len(nouns) == 1:
            object = nouns[0][1]
            part = None
        elif len(nouns) > 1:
            # Assume part comes after
            object = nouns[0][1]
            part = nouns[1][1]
        else:
            raise Exception("Invalid sentence with no target object")
    
    # print('object is', object)

    # Get adjectives of the object
    object_name = ''
    for child in object.children:
        if str(child.dep_) in ('amod', 'compound'):
            object_name += child.text + ' '
    object_name += object.text

    if str(object.dep_) in ('amod', 'compound'):
        object_name += ' ' + object.head.text

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
                holding_parts += dobjs

        # print('holding', holding_parts, 'giving', giving_parts)

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

    return object_name.strip(' '), part, target_holder


def parse_audio(file="gaze_utils/2 Mustard.wav"):
    command = \
    'import whisper;' + \
    'model = whisper.load_model("base");' + \
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
        text = parse_audio('gaze_utils/audio.wav')
    else:
        print('Simulating prompt for 4 seconds')
        time.sleep(4)
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
    record_and_parse()