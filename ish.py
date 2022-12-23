# ish.py

'''
# imap_sorting_hat = "ish"
Magically sort email into smart folders

- No rule programming. Instead, just move a few emails into a smart folder and **ish** will quickly learn what the messages have in common.
- Any folder can be labeled a smart folder.
- Uses the lates OpenAI language model technology to quickly sort emails into corresponding folders.
- Compatible with all imap email clients.
- Works for all common languages.

Status: Early development
'''

import os
import sys
import imaplib
from typing import List, Dict
import logging
from getpass import getpass
import yaml
from os.path import exists, isdir, join
from dataclasses import dataclass
import shelve
import requests

import imapclient
import numpy as np
import openai


logging.basicConfig(level=logging.INFO)


class Settings(dict):
    '''Settings for the application'''
    @staticmethod
    def get_user_directory():
        if os.name == 'nt':
            dir = os.path.expandvars('%USERPROFILE%')
        elif os.name == 'posix':
            dir = os.path.expandvars('$HOME')
        elif os.name == 'mac':
            dir = os.path.expandvars('$HOME')
        return dir

    @property
    def settings_file(self):
        ishd = os.path.join(self.get_user_directory(), '.ish')
        os.makedirs(ishd, exist_ok=True)
        ishfile = os.path.join(ishd, 'settings.yaml')
        return ishfile

    def read(self):
        with open(self.settings_file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.update(data)

    def write(self):
        with open(self.settings_file, 'w') as f:
            yaml.dump(dict(self), f)
            logging.info('Settings saved to %s', self.settings_file)

    def __init__(self):
        self['source_folders'] = []
        self['destination_folders'] = []
        self['ignore_folders'] = []
        self['host'] = ''
        self['username'] = ''
        self['password'] = ''
        self['data_directory'] = ''
        self['openai_api_key'] = ''
        self['openai_model'] = 'text-embedding-ada-002'

        if exists(self.settings_file):
            self.read()

    def update_data_settings(self):
        while not isdir(self['data_directory']):
            if self['data_directory'] == '':
                self['data_directory'] = join(self.get_user_directory(), '.ish', 'data')
            else:
                print(f'Invalid data directory: {self["data_directory"]}')

            self['data_directory'] = input(f'Enter data directory[{self["data_directory"]}]: ').strip() or self['data_directory']
            try:
                os.makedirs(self['data_directory'], exist_ok=True)
            except IOError:
                pass

        self.write()

    def update_openai_settings(self):
        self['openai_api_key'] = input('Enter OpenAI API key (see https://beta.openai.com)): ')
        self.write()

    def update_login_settings(self):
        self.update({
            'host': input(f'Enter imap server [{self["host"]}]: ').strip() or self['host'],
            'username': input(f'Enter username [{self["username"]}]: ').strip() or self['username'],
            'password': getpass('Enter password: ')
        })
        self.write()

    def update_folder_settings(self, folders: List[str]):
        source_folders = set(self['source_folders'])
        destination_folders = set(self['destination_folders'])
        ignore_folders = set(self['ignore_folders'])
        all_folders = source_folders | destination_folders | ignore_folders

        for folder in folders:
            if folder not in all_folders:
                opt = None
                while opt not in ['s', 'd', 'i']:
                    opt = input(f'Folder {folder} is not configured. What do you want to do with it? [s]ource, [d]estination, [i]gnore: ')

                if opt == 's':
                    source_folders.add(folder)
                elif opt == 'd':
                    destination_folders.add(folder)
                elif opt == 'i':
                    ignore_folders.add(folder)

        missing_folders = all_folders - set(folders)
        if missing_folders:
            for folder in missing_folders:
                logging.info(f'Folder {folder} is missing. It will be removed from the settings.')

            source_folders -= missing_folders
            destination_folders -= missing_folders
            ignore_folders -= missing_folders

        self['source_folders'] = sorted(source_folders)
        self['destination_folders'] = sorted(destination_folders)
        self['ignore_folders'] = sorted(ignore_folders)

        self.write()


class ISH:
    def __init__(self) -> None:
        self.settings = Settings()
        self.imap_conn = None
        self.hkey = b'BODY[HEADER.FIELDS (SUBJECT FROM TO CC BCC)]'
        self.bkey = b'BODY[]'

    def connect_imap(self) -> bool:
        if not self.settings['host'] or not self.settings['username']:
            return False

        try:
            imap_conn = imapclient.IMAPClient(self.settings['host'], ssl=True)
            imap_conn.login(self.settings['username'], self.settings['password'])
        except imaplib.IMAP4.error as e:
            logging.error(e)
            return False

        self.imap_conn = imap_conn
        return True

    def connect_openai(self) -> bool:
        if not self.settings['openai_api_key']:
            return False

        openai.api_key = self.settings['openai_api_key']
        # check if api key is valid
        try:
            response = openai.Engine.list()
        except openai.APIError as e:
            logging.error(e)
            return False
        return True

    def configure(self):
        '''Configure ish and connect to imap and openai'''
        settings = self.settings

        self.settings.update_data_settings()

        while not self.connect_imap():
            settings.update_login_settings()

        while not self.connect_openai():
            settings.update_openai_settings()

        folders = [t[2] for t in self.imap_conn.list_folders()]
        settings.update_folder_settings(folders)

        print('Configuration complete')

    def connect_noninteractive(self) -> bool:
        '''Connect to imap and openai'''
        if not self.connect_imap():
            logging.error(f'Failed to connect to imap server. Configure, or check your settings in {self.settings.settings_file}')
            return False

        if not self.connect_openai():
            logging.error(f'Failed to connect to openai. Configure or check your settings in {self.settings.settings_file}')
            return False

        return True

    def get_msgs(self, uids:List[int]) -> Dict[int, str]:
        '''Fetch new messages through cache {uid: 'msg'}'''
        d = {}
        new_uids = []
        filename = join(self.settings['data_directory'], 'msgs.db')

        with shelve.open(filename) as f:
            for uid in uids:
                try:
                    d[uid] = f[str(uid)]
                    continue
                except KeyError:
                    new_uids.append(uid)

            if len(new_uids) > 0:
                msgs = self.imap_conn.fetch(new_uids, [self.hkey, self.bkey])
                for uid in new_uids:
                    d[uid] = msgs[uid][self.hkey].decode('utf-8') + msgs[uid][self.bkey].decode('utf-8')
                    f[str(uid)]  = d[uid]

        return d

    def get_embeddings(self, uids:List[int]) -> Dict[int, np.ndarray]:
        '''Get embeddings using OpenAI API through cache {uid: embedding}'''
        d = {}
        new_uids = []
        filename = join(self.settings['data_directory'], 'embd.db')

        with shelve.open(filename) as f:
            for uid in uids:
                try:
                    d[uid] = f[str(uid)]
                    continue
                except KeyError:
                    new_uids.append(uid)

            if len(new_uids) > 0:
                msgs = self.get_msgs(new_uids)

                for uid, msg in msgs.items():
                    d[uid] = openai.Completion.create(
                        engine=self.settings['openai_model'], prompt=msg, max_tokens=5, stop=['\r\n', '\n'], temperature=0.0,
                        logprobs=10, echo=True, logprobs_token='__logprobs__')['choices'][0]['logprobs']['tokens']
                    f[str(uid)] = d[uid]

        return d

    def process_source_folder(self, folder:str):

        imap_conn = self.imap_conn
        imap_conn.select_folder(folder)

        # Retrieve the UIDs of all messages in the folder
        uids = imap_conn.search(['ALL'])
        msgs = self.get_msgs(uids[:20])

        # Print the subjects of all messages
        for uid in uids[:20]:
            content = msgs[uid]
            print (content[:4096])
            print ('-'*80)


    def process_destination_folder(self, imap_conn, folder:str):
        imap_conn.select_folder(folder)

        # Retrieve the UIDs of all messages in the folder
        uids = imap_conn.search(['ALL'])

        # Print the subjects of all messages
        for uid in uids[:20]:
            raw_msg = imap_conn.fetch([uid], [b'BODY[HEADER.FIELDS (SUBJECT FROM TO CC BCC)]', b'BODY[]'])
            content = raw_msg[uid][b'BODY[HEADER.FIELDS (SUBJECT FROM TO CC BCC)]'].decode('utf-8') + raw_msg[uid][b'BODY[]'].decode('utf-8')

            print (content[:4096])
            print ('-'*80)

    def run(self, interactive:bool) -> int:
        if interactive:
            self.configure()
        else:
            if not self.connect_noninteractive():
                return 1

        settings = self.settings

        for f in settings['source_folders']:
            logging.info(f'Source folder: {f}')

        for f in settings['destination_folders']:
            logging.info(f'Destination folder: {f}')

        for f in settings['source_folders']:
            self.process_source_folder(f)

        #for f in settings['destination_folders']:
        #    self.process_destination_folder(imap_conn, f)

        self.imap_conn.logout()
        return 0

def main():
    ish = ISH()
    r = ish.run(interactive=True)
    sys.exit(r)

if __name__ == '__main__':
    main()

