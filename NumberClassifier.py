#check if a token is a Roman number
def IsRomanNumeral(token):
    return bool(re.match(r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
, token))

print(IsRomanNumeral("MCDXXXVII"))

Numbers ={
    'zero', 
    'one', 'once','first',
    'two', 'twice','second',
    'three', 'third', 'triple',
    'four', 'fourth', 'forth','quarter',
    'five', 'fifth',
    'six', 'sixth',
    'seven','seventh',
    'eight','eighth',
    'nine','ninth',
    'ten','tenth',
    'eleven','eleventh',
    'twelve','twelfth',
    'thirteen','thirteenth',
    'fourteen','fourteenth',
    'fifteen','fifteenth',
    'sixteen','sixteenth',
    'seventeen','seventeenth',
    'eighteen','eighteenth',
    'nineteen','nineteenth',
    'twenty','twentieth',
    'thirty','thirtieth',
    'forty','fourtieth','fortieth',
    'fifty','fiftieth',
    'sixty','sixtieth',
    'seventy','seventieth',
    'eighty','eightieth',
    'ninty','nintieth','ninety','ninetieth',
    'hundred','hundredth','century',
    'thousand',
    'million',
    'billion',
    'trillion',
    'millennium',
    'byte'
}

from dateutil.parser import parse

def is_date(string):
    try: 
        parse(string)
        return True
    except ValueError:
        return False
print("is_date", is_date("Mon"))
print("is_date", is_date("13th"))
print("is_date", is_date("september"))
print("is_date", is_date("12:00:00"))
print("is_date", is_date("12am"))
def is_number(s): #A basic function to check if a word/token is a number or not
    try:
        float(s)
        return True
    except ValueError:
        # print(s)
        if s.lower() in Numbers or IsRomanNumeral(s):
            return True
        elif s[len(s)-1] == u'%' and is_number(s[:len(s)-1]):
            return True
        else:
            return False
print("is_number",is_number("2e5"))
print("is_number",is_number("quarter"))
print("is_number",is_number("22%"))
