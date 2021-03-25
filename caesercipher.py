# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 20:24:36 2020

@author: hycbl
"""
shift = 3 # defining the shift count
text = "HELLO WORLD"
encryption = ""
for c in text:
    # check if character is an uppercase letter
    if c.isupper():
        # find the position in 0-25
        c_unicode = ord(c)
        c_index = ord(c) - ord("A")
        # perform the shift
        new_index = (c_index + shift) % 26
        # convert to new character
        new_unicode = new_index + ord("A")
        new_character = chr(new_unicode)
        # append to encrypted string
        encryption = encryption + new_character
    else:
        # since character is not uppercase, leave it as it is
        encryption += c        
print("Plain text:",text)
print("Encrypted text:",encryption)

########################################################################

shift = 3 # defining the shift count
encrypted_text = "KHOOR ZRUOG"
plain_text = ""
for c in encrypted_text:
    # check if character is an uppercase letter
    if c.isupper():
        # find the position in 0-25
        c_unicode = ord(c)
        c_index = ord(c) - ord("A")
        # perform the negative shift
        new_index = (c_index - shift) % 26
        # convert to new character
        new_unicode = new_index + ord("A")
        new_character = chr(new_unicode)
        # append to plain string
        plain_text = plain_text + new_character
    else:
        # since character is not uppercase, leave it as it is
        plain_text += c
print("Encrypted text:",encrypted_text)
print("Decrypted text:",plain_text)

######################################################################

# The Encryption Function
def cipher_encrypt(plain_text, key):
    encrypted = ""
    for c in plain_text:
        if c.isupper(): #check if it's an uppercase character
            c_index = ord(c) - ord('A')
            # shift the current character by key positions
            c_shifted = (c_index + key) % 26 + ord('A')
            c_new = chr(c_shifted)
            encrypted += c_new
        elif c.islower(): #check if its a lowecase character
            # subtract the unicode of 'a' to get index in [0-25) range
            c_index = ord(c) - ord('a') 
            c_shifted = (c_index + key) % 26 + ord('a')
            c_new = chr(c_shifted)
            encrypted += c_new
        elif c.isdigit():
            # if it's a number,shift its actual value 
            c_new = (int(c) + key) % 10
            encrypted += str(c_new)
        else:
            # if its neither alphabetical nor a number, just leave it like that
            encrypted += c
    return encrypted
# The Decryption Function
def cipher_decrypt(ciphertext, key):
    decrypted = ""
    for c in ciphertext:
        if c.isupper(): 
            c_index = ord(c) - ord('A')
            # shift the current character to left by key positions to get its original position
            c_og_pos = (c_index - key) % 26 + ord('A')
            c_og = chr(c_og_pos)
            decrypted += c_og

        elif c.islower(): 
            c_index = ord(c) - ord('a') 
            c_og_pos = (c_index - key) % 26 + ord('a')
            c_og = chr(c_og_pos)
            decrypted += c_og
        elif c.isdigit():
            # if it's a number,shift its actual value 
            c_og = (int(c) - key) % 10
            decrypted += str(c_og)
        else:
            # if its neither alphabetical nor a number, just leave it like that
            decrypted += c
    return decrypted


plain_text = "Mate, the adventure ride in Canberra was so much fun, We were so drunk we ended up calling 911!"
ciphertext = cipher_encrypt(plain_text, 4)
print("Plain text message:\n", plain_text)
print("Encrypted ciphertext:\n", ciphertext)

###########################################################

table = str.maketrans("abcde", "01234")
text = "Albert Einstein, born in Germany, was a prominent theoretical physicist."
translated = text.translate(table)
print("Original text:/n", text)
print("Translated text:/n", translated)



import string
def cipher_cipher_using_lookup(text,  key, characters = string.ascii_lowercase, decrypt=False):
    if key < 0:
        print("key cannot be negative")
        return None
    n = len(characters)
    if decrypt==True:
        key = n - key
    table = str.maketrans(characters, characters[key:]+characters[:key])    
    translated_text = text.translate(table)    
    return translated_text


text = "HELLO WORLD! Welcome to the world of Cryptography!"
encrypted = cipher_cipher_using_lookup(text, 3, string.ascii_uppercase, decrypt=False)
print(encrypted)




character_set = string.ascii_lowercase + string.ascii_uppercase + string.digits + " "+ string.punctuation
print("Extended character set:\n", character_set)
plain_text = "My name is Dave Adams. I am living on the 99th street. Please send the supplies!"
encrypted = cipher_cipher_using_lookup(plain_text, 5, character_set, decrypt=False)
print("Plain text:\n", plain_text)
print("Encrypted text:\n", encrypted)


##########################################################

import string
def cipher_cipher_using_lookup(text, key, characters = string.ascii_lowercase, decrypt=False, shift_type="right"):
    if key < 0:
        print("key cannot be negative")
        return None

    n = len(characters)
    if decrypt==True:
        key = n - key

    if shift_type=="left":
        # if left shift is desired, we simply inverse they sign of the key
        key = -key
    table = str.maketrans(characters, characters[key:]+characters[:key])
    translated_text = text.translate(table)
    return translated_text



text = "Hello World !"
encrypted = cipher_cipher_using_lookup(text, 3, characters = (string.ascii_lowercase + string.ascii_uppercase), decrypt = False, shift_type="left")
print("plain text:", text)
print("encrypted text with negative shift:",encrypted)



######################################################################

def fileCipher(fileName, outputFileName, key = 3, shift_type = "right", decrypt=False):
    with open(fileName, "r") as f_in:
        with open(outputFileName, "w") as f_out:
            # iterate over each line in input file
            for line in f_in:
                #encrypt/decrypt the line
                lineNew = cipher_cipher_using_lookup(line, key, decrypt=decrypt, shift_type=shift_type)
                #write the new line to output file
                f_out.write(lineNew)                    
    print("The file {} has been translated successfully and saved to {}".format(fileName, outputFileName))

inputFile = "./sth.txt"
outputFile = "./sth_encrypted.txt"
fileCipher(inputFile, outputFile, key=3, shift_type="right", decrypt = False)


########################################################################


def vigenere_cipher(text, keys, decrypt=False):
    # vigenere cipher for lowercase letters
    n = len(keys)
    translatedText =""
    i = 0 #used to record the count of lowercase characters processed so far
    # iterate over each character in the text
    for c in text:
        #translate only if c is lowercase
        if c.islower():
            shift = keys[i%n] #decide which key is to be used
            if decrypt == True:
                # if decryption is to be performed, make the key negative
                shift = -shift
            # Perform the shift operation
            shifted_c = chr((ord(c) - ord('a') + shift)%26 + ord('a'))
            translatedText += shifted_c
            i += 1
        else:
            translatedText += c            
    return translatedText


text = "we will call the first manned moon mission the Project Apollo"
encrypted_text = vigenere_cipher(text, [1,2,3])
print("Plain text:\n", text)
print("Encrypted text:\n", encrypted_text)


##########################################################################

def cipher_decrypt_lower(ciphertext, key):
    decrypted = ""
    for c in ciphertext:
        if c.islower(): 
            c_index = ord(c) - ord('a') 
            c_og_pos = (c_index - key) % 26 + ord('a')
            c_og = chr(c_og_pos)
            decrypted += c_og
        else:
            decrypted += c
    return decrypted


cryptic_text = "ks gvozz ohhoqy hvsa tfca hvs tfcbh oh bccb cb Tisgrom"
for i in range(0,26):
    plain_text = cipher_decrypt_lower(cryptic_text, i)
    print("For key {}, decrypted text: {}".format(i, plain_text))













































































