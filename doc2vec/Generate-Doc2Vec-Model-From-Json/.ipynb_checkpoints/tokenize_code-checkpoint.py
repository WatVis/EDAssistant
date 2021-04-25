import tokenize
from io import BytesIO

def tokenize_code(code, cell_type):
    # if markdown or raw, split with " "
    if cell_type != "code":
        return []
    
    
    tokenized_code = []
    tokens = []
    
    try:
        tokens = tokenize.tokenize(BytesIO(code.encode()).readline)
    except (SyntaxError, tokenize.TokenError, IndentationError, AttributeError):
        return []
    try:
#       tokens is a generator function, so we need to also catch exceptions when calling it
        for tok in tokens:
            ret = ""
            # first token is always utf-8, ignore it
            if (tok.string == "utf-8"):
                continue
            # type 4 is NEWLINE
            elif (tok.type == 4 or tok.type == 61):
                ret = "[NEWLINE]"
            # type 5 is INDENT
            elif (tok.type == 5):
                ret = "[INDENT]"
            else:
                ret = tok.string
    #         print(tok)
    #         print(f"Type: {tok.exact_type}\nString: {tok.string}\nStart: {tok.start}\nEnd: {tok.end}\nLine: {tok.line.strip()}\n======\n")
            tokenized_code.append(ret)
        return tokenized_code
    except (SyntaxError, tokenize.TokenError, IndentationError, AttributeError):
        return []