import re


sentence = "!@ssdks#$@  ncsjfja  $@$(@*%&%  dfkd  jfkd  j><??   "

my_pattern = re.compile(r'[!@#$%^&*(-),.?;:"><]')

sentence_test = re.sub(my_pattern, "", sentence)
print(sentence_test)
print(len(sentence_test))
sentence_test = sentence_test.strip()
print(sentence_test)
print(len(sentence_test))
sentence_test = sentence_test.replace(" ", "")
print(sentence_test)
print()