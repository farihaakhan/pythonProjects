# -----------------------------------------------------------------------------
# calc.py
#
# A simple calculator with variables.  Asks the user for more input and
# demonstrates the use of the t_eof() rule.
# -----------------------------------------------------------------------------

# https://replit.com/@Farihak/HW3

import sys

sys.path.insert(0, "../..")

tokens = (
    'NAME', 'NUMBER', 'EQUALITY', 'TERNARY'
)

literals = ['=', '+', '-', '*', '/', '(', ')']

# Tokens

t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'
t_EQUALITY = r'=='
t_TERNARY = r':'


def t_NUMBER(t):
    r'\d+\.\d+'
    t.value = float(t.value)
    return t


t_ignore = " \t"


def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")


def t_eof(t):
    more = input('... ')
    if more:
        t.lexer.input(more + '\n')
        return t.lexer.token()
    else:
        return None


def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


# Build the lexer
import ply.lex as lex

lex.lex()

# Parsing rules

precedence = (
    ('left', '+', '-'),
    ('left', '*', '/'),
    ('right', 'UMINUS'),
    ('left', 'TERNARY')
)

# dictionary of names
names = {}


def p_statement_assign(p):
    'statement : NAME "=" expression'
    names[p[1]] = p[3]


def p_statement_expr(p):
    'statement : expression'
    print(p[1])


def p_expression_binop(p):
    '''expression : expression '+' expression
                  | expression '-' expression
                  | expression '*' expression
                  | expression '/' expression
                  | expression EQUALITY expression
  '''

    if p[2] == '+':
        p[0] = p[1] + p[3]
    elif p[2] == '-':
        p[0] = p[1] - p[3]
    elif p[2] == '*':
        p[0] = p[1] * p[3]
    elif p[2] == '/':
        p[0] = p[1] / p[3]
    elif p[2] == '==':
        if p[1] == p[3]:
            p[0] = 1.0
        else:
            p[0] = 0.0


def p_expression_terop(p):
    '''expression : expression '?' expression TERNARY expression

  '''
    if p[1] == '?' and p[3] == ':':
        if p[0] != 0.0:
            print(p[2])
        else:
            print(p[4])


def p_expression_uminus(p):
    "expression : '-' expression %prec UMINUS"
    p[0] = -p[2]


def p_expression_group(p):
    "expression : '(' expression ')'"
    p[0] = p[2]


def p_expression_number(p):
    "expression : NUMBER"
    p[0] = p[1]


def p_expression_name(p):
    "expression : NAME"
    try:
        p[0] = names[p[1]]

    except LookupError:
        print("Undefined name '%s'" % p[1])
        p[0] = 0


def p_error(p):
    if p:
        print("Syntax error at '%s'" % p.value)
    else:
        print("Syntax error at EOF")


import ply.yacc as yacc

yacc.yacc()

while True:
    try:
        s = input('calc > ')
    except EOFError:
        break
    if not s:
        continue
    yacc.parse(s + '\n')