import re

def clean_text(text):
    text = text.replace(u'\xa0', u' ')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n +', '\n', text)
    text = re.sub(r'\r?\n(?!\r?\n)', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'\n\n ?', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'- ?\n\n', '- ', text)
    text = re.sub(r'([,\)\w°])( ?\n\n ?)(\w)', '\g<1> \g<3>', text, flags=re.IGNORECASE)
    text = re.sub(r'[0-9]\n\n,', ',', text)
    text = re.sub(r'\n\n([.,]) ?\n\n', '\g<1>\n\n', text)
    text = re.sub(r'\n\n([.,–:]) ?', '\g<1> ', text)
    text = re.sub(r'\( ?\n\n', '(', text)
    text = re.sub(r'(\w)\n\n(\w)', '\g<1> \g<2>', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\n(["§])\n\n', ' \g<1> ', text)
    text = text.strip('\n\n')

    return text