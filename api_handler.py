from pprint import pprint
import requests
import json

def delete_voices(name):
    geturl = "https://api.elevenlabs.io/v1/voices/"
    delurl = "https://api.elevenlabs.io/v1/voices/"

    ''' CURL REQUEST
    curl -X 'GET' \       
      'https://api.elevenlabs.io/v1/voices' \
      -H 'accept: application/json' \
      -H 'xi-api-key: b967f41c8903f6c60311efce2b16c470'|jq
    '''
    # Processing training data and creating txt file corpus
    with open("apikey3.txt", 'r') as f:
        API_KEY = f.read()

    headers = {
      "Accept": "application/json",
      "xi-api-key": API_KEY
    }

    ids = []

    getresp = json.loads(requests.get(geturl, headers=headers).text)
    #pprint(getresp['voices'])
    for voicedict in getresp['voices']:
        if voicedict['name'] == name:
            #print(voicedict['name'])
            ids.append(voicedict['voice_id'])

    # print(ids)
    for id in ids:
        delresp = requests.delete(delurl + id, headers=headers)
        print(delresp)

    #delresp = requests.delete(url, headers=headers)

    print("Deleted all instances of " + name + ".")


def print_voices():
    geturl = "https://api.elevenlabs.io/v1/voices/"
    delurl = "https://api.elevenlabs.io/v1/voices/"

    ''' CURL REQUEST
    curl -X 'GET' \       
      'https://api.elevenlabs.io/v1/voices' \
      -H 'accept: application/json' \
      -H 'xi-api-key: b967f41c8903f6c60311efce2b16c470'|jq
    '''
    # Processing training data and creating txt file corpus
    with open("apikey3.txt", 'r') as f:
        API_KEY = f.read()

    headers = {
      "Accept": "application/json",
      "xi-api-key": API_KEY
    }

    getresp = json.loads(requests.get(geturl, headers=headers).text)
    #pprint(getresp['voices'])
    for voicedict in getresp['voices']:
        if "Nixon" == voicedict['name'][:5]:
            print(voicedict['name'])


if __name__ == "__main__":
    print_voices()