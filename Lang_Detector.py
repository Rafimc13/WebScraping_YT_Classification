import re


class LangDetect:
    # Separating the Greek/English/other languages. If we have at least 3 greek chars is a greek title-comment
    pat_gr = re.compile(r'[α-ωΑ-Ωίϊΐόύϋΰέώ]{3}')
    pat_eng = re.compile(r'[a-zA-Z]')
    # Creating patterns that are written with English char, but they definitely have Greek meaning
    pat_greeklish = re.compile(r'\b(egw|esi|emeis|esy|eseis|afoy|mou|moy|sou|soy|afou|autoi|autes|aytoi|geia|apo|ton|'
                               r'gia|kai|ti|ta|tis|tous|tou|toys|oi|o|h|autos|auta|oti|na|nai|oxi|den|tha|8a)\b', flags=re.IGNORECASE)

    def read_txt(self, txt):
        ls_sentences = []
        with open(txt, 'r', encoding='utf-8') as file:
            for line in file:
                line.strip()
                ls_sentences.append(line)

        return ls_sentences

    def pattern_search(self, sentence, pattern=None):
        if pattern is None:
            pattern = re.compile(r'\.\n|(\d{1,2}\. )')
        try:
            sentence = re.sub(pattern, "", sentence)
        except Exception as e:
            print(f"You have an error: {e}")

        return sentence

    def comp_languages(self, sentence, languages):
        lang = None
        if re.search(self.pat_gr, sentence):
            lang = languages[0]
        elif re.search(self.pat_eng, sentence):
            count_eng = len(re.findall(self.pat_eng, sentence))
            my_pattern = re.compile(r'[!@#$%^&*(-),.?;:"><]')
            clean_sentence = self.pattern_search(sentence, my_pattern)
            clean_sentence = clean_sentence.replace(" ", "")
            if count_eng < 0.99 * len(clean_sentence):
                lang = languages[3]
            else:
                lang = languages[2]

        if lang == languages[2] or lang == languages[3]:
            if re.search(self.pat_greeklish, sentence):
                lang = languages[1]

        return lang

    def comp_scores(self, dataset, column1, column2):
        my_sum = 0
        for i in range(len(dataset)):
            if dataset.iloc[i][column1] == dataset.iloc[i][column2]:
                my_sum += 1
        score = my_sum / len(dataset)
        print(f'Accuracy of my language detector is: {score * 100}%')
        return score
