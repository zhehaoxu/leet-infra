"""Frozen regex 2026.9.29 / tokenizers 0.23.3 fixtures; provenance is documented in notes."""


# Index is a byte, value is its encoded Unicode character: all 256 entries.
_BYTE_ALPHABET = (
    '\u0100\u0101\u0102\u0103\u0104\u0105\u0106\u0107\u0108\u0109\u010a\u010b\u010c\u010d\u010e\u010f'
    '\u0110\u0111\u0112\u0113\u0114\u0115\u0116\u0117\u0118\u0119\u011a\u011b\u011c\u011d\u011e\u011f\u0120'
    '!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
    '\u0121\u0122\u0123\u0124\u0125\u0126\u0127\u0128\u0129\u012a\u012b\u012c\u012d\u012e\u012f'
    '\u0130\u0131\u0132\u0133\u0134\u0135\u0136\u0137\u0138\u0139\u013a\u013b\u013c\u013d\u013e\u013f\u0140\u0141\u0142'
    '\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\u0143\xae\xaf'
    '\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf'
    '\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf'
    '\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf'
    '\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef'
    '\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff'
)
_INVERSE = {char: byte for byte, char in enumerate(_BYTE_ALPHABET)}

_FIXTURES = {
    'empty': [
        ('', []),
        ('a', [('a', 0, 1)]),
    ],
    'contractions': [
        ("'s't're've'm'll'd", [("'s", 0, 2), ("'t", 2, 4), ("'re", 4, 7), ("'ve", 7, 10), ("'m", 10, 12), ("'ll", 12, 15), ("'d", 15, 17)]),
        ("We're I've I'm I'll I'd can't", [('We', 0, 2), ("'re", 2, 5), ('\u0120I', 5, 7), ("'ve", 7, 10), ('\u0120I', 10, 12), ("'m", 12, 14), ('\u0120I', 14, 16), ("'ll", 16, 19), ('\u0120I', 19, 21), ("'d", 21, 23), ('\u0120can', 23, 27), ("'t", 27, 29)]),
    ],
    'case': [
        ("'S'T'RE'VE'M'LL'D", [("'", 0, 1), ('S', 1, 2), ("'", 2, 3), ('T', 3, 4), ("'", 4, 5), ('RE', 5, 7), ("'", 7, 8), ('VE', 8, 10), ("'", 10, 11), ('M', 11, 12), ("'", 12, 13), ('LL', 13, 15), ("'", 15, 16), ('D', 16, 17)]),
        ("He's HE'S he\u2019 s", [('He', 0, 2), ("'s", 2, 4), ('\u0120HE', 4, 7), ("'", 7, 8), ('S', 8, 9), ('\u0120he', 9, 12), ('\xe2\u0122\u013b', 12, 13), ('\u0120s', 13, 15)]),
    ],
    'punctuation': [
        ("!!'s 's", [("!!'", 0, 3), ('s', 3, 4), ("\u0120'", 4, 6), ('s', 6, 7)]),
        ("!? 're x's!!'t", [('!?', 0, 2), ("\u0120'", 2, 4), ('re', 4, 6), ('\u0120x', 6, 8), ("'s", 8, 10), ("!!'", 10, 13), ('t', 13, 14)]),
        ('...123--abc_42', [('...', 0, 3), ('123', 3, 6), ('--', 6, 8), ('abc', 8, 11), ('_', 11, 12), ('42', 12, 14)]),
        (" 's 're", [("\u0120'", 0, 2), ('s', 2, 3), ("\u0120'", 3, 5), ('re', 5, 7)]),
    ],
    'letters': [
        ('\u4f60\u597d\u4e16\u754c \u039a\u03b1\u03bb\u03b7\u03bc\u03ad\u03c1\u03b1 \u0416\u0438\u0437\u043d\u044c caf\xe9', [('\xe4\xbd\u0142\xe5\xa5\xbd\xe4\xb8\u0138\xe7\u0137\u012e', 0, 4), ('\u0120\xce\u013c\xce\xb1\xce\xbb\xce\xb7\xce\xbc\xce\u0143\xcf\u0123\xce\xb1', 4, 13), ('\u0120\xd0\u0138\xd0\xb8\xd0\xb7\xd0\xbd\xd1\u012e', 13, 19), ('\u0120caf\xc3\xa9', 19, 24)]),
        ('\xe9 e\u0301 \xc5 A\u030a', [('\xc3\xa9', 0, 1), ('\u0120e', 1, 3), ('\xcc\u0123', 3, 4), ('\u0120\xc3\u0127', 4, 6), ('\u0120A', 6, 8), ('\xcc\u012c', 8, 9)]),
        ('\u2163\xb2\u0663A\xaa', [('\xe2\u0127\xa3\xc2\xb2\xd9\xa3', 0, 3), ('A\xc2\xaa', 3, 5)]),
    ],
    'numbers': [
        ('42 \u0663\u0664 \u0967\u0968 \u2163 \xb2 \xbd', [('42', 0, 2), ('\u0120\xd9\xa3\xd9\xa4', 2, 5), ('\u0120\xe0\xa5\xa7\xe0\xa5\xa8', 5, 8), ('\u0120\xe2\u0127\xa3', 8, 10), ('\u0120\xc2\xb2', 10, 12), ('\u0120\xc2\xbd', 12, 14)]),
        ('a1b\xb2c\u216b\u2169', [('a', 0, 1), ('1', 1, 2), ('b', 2, 3), ('\xc2\xb2', 3, 4), ('c', 4, 5), ('\xe2\u0127\xab\xe2\u0127\xa9', 5, 7)]),
        ('\U0001d7d8\uff19 1.2e-3', [('\xf0\u013f\u0141\u013a\xef\xbc\u013b', 0, 2), ('\u01201', 2, 4), ('.', 4, 5), ('2', 5, 6), ('e', 6, 7), ('-', 7, 8), ('3', 8, 9)]),
    ],
    'marks_emoji': [
        ('a\u0301\u0327b', [('a', 0, 1), ('\xcc\u0123\xcc\xa7', 1, 3), ('b', 3, 4)]),
        ('\U0001f469\u200d\U0001f4bb \U0001f1e8\U0001f1f3 \U0001f44d\U0001f3fd', [('\xf0\u0141\u0133\xa9\xe2\u0122\u012f\xf0\u0141\u0134\xbb', 0, 3), ('\u0120\xf0\u0141\u0129\xa8\xf0\u0141\u0129\xb3', 3, 6), ('\u0120\xf0\u0141\u0133\u012f\xf0\u0141\u0131\xbd', 6, 9)]),
        ('\u2764\ufe0f\u2665\ufe0f!', [('\xe2\u013f\xa4\xef\xb8\u0131\xe2\u013b\xa5\xef\xb8\u0131!', 0, 5)]),
    ],
    'spaces': [
        ('a  b   c ', [('a', 0, 1), ('\u0120', 1, 2), ('\u0120b', 2, 4), ('\u0120\u0120', 4, 6), ('\u0120c', 6, 8), ('\u0120', 8, 9)]),
        ('  a    ', [('\u0120', 0, 1), ('\u0120a', 1, 3), ('\u0120\u0120\u0120\u0120', 3, 7)]),
        ('a\t\t!', [('a', 0, 1), ('\u0109', 1, 2), ('\u0109', 2, 3), ('!', 3, 4)]),
        ('a\n b\r\n\t c', [('a', 0, 1), ('\u010a', 1, 2), ('\u0120b', 2, 4), ('\u010d\u010a\u0109', 4, 7), ('\u0120c', 7, 9)]),
    ],
    'whitespace': [
        ('a\xa0b\x85c', [('a', 0, 1), ('\xc2\u0142', 1, 2), ('b', 2, 3), ('\xc2\u0127', 3, 4), ('c', 4, 5)]),
        ('\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000', [('\u0109\u010a\u010b\u010c\u010d\u0120\xc2\u0127\xc2\u0142\xe1\u013c\u0122\xe2\u0122\u0122\xe2\u0122\u0123\xe2\u0122\u0124\xe2\u0122\u0125\xe2\u0122\u0126\xe2\u0122\u0127\xe2\u0122\u0128\xe2\u0122\u0129\xe2\u0122\u012a\xe2\u0122\u012b\xe2\u0122\u012c\xe2\u0122\xa8\xe2\u0122\xa9\xe2\u0122\xaf\xe2\u0123\u0141\xe3\u0122\u0122', 0, 25)]),
        ('a\u2028\u2029 b', [('a', 0, 1), ('\xe2\u0122\xa8\xe2\u0122\xa9', 1, 3), ('\u0120b', 3, 5)]),
        (' \xa0x\xa0 y', [('\u0120', 0, 1), ('\xc2\u0142', 1, 2), ('x', 2, 3), ('\xc2\u0142', 3, 4), ('\u0120y', 4, 6)]),
    ],
    'controls': [
        ('a\x1cb\x1dc\x1ed\x1fe', [('a', 0, 1), ('\u011c', 1, 2), ('b', 2, 3), ('\u011d', 3, 4), ('c', 4, 5), ('\u011e', 5, 6), ('d', 6, 7), ('\u011f', 7, 8), ('e', 8, 9)]),
        ('\x00\x01\x08\x7f', [('\u0100\u0101\u0108\u0121', 0, 4)]),
        ('\u200b\u200c\u200d\ufeff', [('\xe2\u0122\u012d\xe2\u0122\u012e\xe2\u0122\u012f\xef\xbb\xbf', 0, 4)]),
        ('\x1c\x1d!!\x1e\x1f', [('\u011c\u011d!!\u011e\u011f', 0, 6)]),
    ],
    'offsets': [
        ('\u4f60\u597d  caf\xe9\n', [('\xe4\xbd\u0142\xe5\xa5\xbd', 0, 2), ('\u0120', 2, 3), ('\u0120caf\xc3\xa9', 3, 8), ('\u010a', 8, 9)]),
        ('a\U0001f600\xe9 \U0001d7d8!', [('a', 0, 1), ('\xf0\u0141\u013a\u0122', 1, 2), ('\xc3\xa9', 2, 3), ('\u0120\xf0\u013f\u0141\u013a', 3, 5), ('!', 5, 6)]),
        ('\U00010000\U000e0001\U000f0000\U00100000', [('\xf0\u0132\u0122\u0122', 0, 1), ('\xf3\u0142\u0122\u0123\xf3\xb0\u0122\u0122\xf4\u0122\u0122\u0122', 1, 4)]),
    ],
    'random': [
        ("\u0301\u030199'", [('\xcc\u0123\xcc\u0123', 0, 2), ('99', 2, 4), ("'", 4, 5)]),
        ('\n_\u0663 \u2028a\u0301AZ\x1c9!', [('\u010a', 0, 1), ('_', 1, 2), ('\xd9\xa3', 2, 3), ('\u0120', 3, 4), ('\xe2\u0122\xa8', 4, 5), ('a', 5, 6), ('\xcc\u0123', 6, 7), ('AZ', 7, 9), ('\u011c', 9, 10), ('9', 10, 11), ('!', 11, 12)]),
        ("\xa0a\xe9\u200d9!\u21630\u4e2d\xe9'\u4e2d0\u2028\u4e2da?0\x85bb \u2163\u0416\u2163\u0301\u0663\u0663\u0416_!", [('\xc2\u0142', 0, 1), ('a\xc3\xa9', 1, 3), ('\xe2\u0122\u012f', 3, 4), ('9', 4, 5), ('!', 5, 6), ('\xe2\u0127\xa30', 6, 8), ('\xe4\xb8\u0143\xc3\xa9', 8, 10), ("'", 10, 11), ('\xe4\xb8\u0143', 11, 12), ('0', 12, 13), ('\xe2\u0122\xa8', 13, 14), ('\xe4\xb8\u0143a', 14, 16), ('?', 16, 17), ('0', 17, 18), ('\xc2\u0127', 18, 19), ('bb', 19, 21), ('\u0120\xe2\u0127\xa3', 21, 23), ('\xd0\u0138', 23, 24), ('\xe2\u0127\xa3', 24, 25), ('\xcc\u0123', 25, 26), ('\xd9\xa3\xd9\xa3', 26, 28), ('\xd0\u0138', 28, 29), ('_!', 29, 31)]),
        ('0\x1ca \xa0\u200db\r\x85a\u03bb0\u0416\x1c_-9\U0001f600\u0416?\u200d\x85\u4e2d ', [('0', 0, 1), ('\u011c', 1, 2), ('a', 2, 3), ('\u0120', 3, 4), ('\xc2\u0142', 4, 5), ('\xe2\u0122\u012f', 5, 6), ('b', 6, 7), ('\u010d', 7, 8), ('\xc2\u0127', 8, 9), ('a\xce\xbb', 9, 11), ('0', 11, 12), ('\xd0\u0138', 12, 13), ('\u011c_-', 13, 16), ('9', 16, 17), ('\xf0\u0141\u013a\u0122', 17, 18), ('\xd0\u0138', 18, 19), ('?\xe2\u0122\u012f', 19, 21), ('\xc2\u0127', 21, 22), ('\xe4\xb8\u0143', 22, 23), ('\u0120', 23, 24)]),
        ('\r1\U0001f600\xb2\t\n?AZA\u0301', [('\u010d', 0, 1), ('1', 1, 2), ('\xf0\u0141\u013a\u0122', 2, 3), ('\xc2\xb2', 3, 4), ('\u0109', 4, 5), ('\u010a', 5, 6), ('?', 6, 7), ('AZA', 7, 10), ('\xcc\u0123', 10, 11)]),
        ("'-!\x1c\u2028a\n\u0416a\u06631\x1c\u2163\r\U0001f600\u200d\u2163\u0663\xe9\u200d9\u2163\xa0", [("'-!\u011c", 0, 4), ('\xe2\u0122\xa8', 4, 5), ('a', 5, 6), ('\u010a', 6, 7), ('\xd0\u0138a', 7, 9), ('\xd9\xa31', 9, 11), ('\u011c', 11, 12), ('\xe2\u0127\xa3', 12, 13), ('\u010d', 13, 14), ('\xf0\u0141\u013a\u0122\xe2\u0122\u012f', 14, 16), ('\xe2\u0127\xa3\xd9\xa3', 16, 18), ('\xc3\xa9', 18, 19), ('\xe2\u0122\u012f', 19, 20), ('9\xe2\u0127\xa3', 20, 22), ('\xc2\u0142', 22, 23)]),
        ("\xa0\u4e2d\xb2\u0301\u03bb\xb2_'\u04160A\u4e2dA_A'\xe9\u21630\xb29\U0001f600\xe9\U0001f600\u0663", [('\xc2\u0142', 0, 1), ('\xe4\xb8\u0143', 1, 2), ('\xc2\xb2', 2, 3), ('\xcc\u0123', 3, 4), ('\xce\xbb', 4, 5), ('\xc2\xb2', 5, 6), ("_'", 6, 8), ('\xd0\u0138', 8, 9), ('0', 9, 10), ('A\xe4\xb8\u0143A', 10, 13), ('_', 13, 14), ('A', 14, 15), ("'", 15, 16), ('\xc3\xa9', 16, 17), ('\xe2\u0127\xa30\xc2\xb29', 17, 21), ('\xf0\u0141\u013a\u0122', 21, 22), ('\xc3\xa9', 22, 23), ('\xf0\u0141\u013a\u0122', 23, 24), ('\xd9\xa3', 24, 25)]),
        ('\xa0\u0301\xb2_\u2028\r\u0301!\x1c \x85Z0\u0663A\xe9\u0663\xa0\u0301A1', [('\xc2\u0142', 0, 1), ('\xcc\u0123', 1, 2), ('\xc2\xb2', 2, 3), ('_', 3, 4), ('\xe2\u0122\xa8', 4, 5), ('\u010d', 5, 6), ('\xcc\u0123!\u011c', 6, 9), ('\u0120', 9, 10), ('\xc2\u0127', 10, 11), ('Z', 11, 12), ('0\xd9\xa3', 12, 14), ('A\xc3\xa9', 14, 16), ('\xd9\xa3', 16, 17), ('\xc2\u0142', 17, 18), ('\xcc\u0123', 18, 19), ('A', 19, 20), ('1', 20, 21)]),
        ("\u0663\xa0\u0416\u04160\rA'\u4e2d  \u03bb\x85\u4e2d1_\u0663\u2163\xe9", [('\xd9\xa3', 0, 1), ('\xc2\u0142', 1, 2), ('\xd0\u0138\xd0\u0138', 2, 4), ('0', 4, 5), ('\u010d', 5, 6), ('A', 6, 7), ("'", 7, 8), ('\xe4\xb8\u0143', 8, 9), ('\u0120', 9, 10), ('\u0120\xce\xbb', 10, 12), ('\xc2\u0127', 12, 13), ('\xe4\xb8\u0143', 13, 14), ('1', 14, 15), ('_', 15, 16), ('\xd9\xa3\xe2\u0127\xa3', 16, 18), ('\xc3\xa9', 18, 19)]),
        ('0\U0001f600\t?Z \t \u4e2d\xb2\U0001f600\xe9\xa0\xb2\xb2\xa0\u0663\u4e2d1\u200d\u0663\u0301\u0416-', [('0', 0, 1), ('\xf0\u0141\u013a\u0122', 1, 2), ('\u0109', 2, 3), ('?', 3, 4), ('Z', 4, 5), ('\u0120\u0109', 5, 7), ('\u0120\xe4\xb8\u0143', 7, 9), ('\xc2\xb2', 9, 10), ('\xf0\u0141\u013a\u0122', 10, 11), ('\xc3\xa9', 11, 12), ('\xc2\u0142', 12, 13), ('\xc2\xb2\xc2\xb2', 13, 15), ('\xc2\u0142', 15, 16), ('\xd9\xa3', 16, 17), ('\xe4\xb8\u0143', 17, 18), ('1', 18, 19), ('\xe2\u0122\u012f', 19, 20), ('\xd9\xa3', 20, 21), ('\xcc\u0123', 21, 22), ('\xd0\u0138', 22, 23), ('-', 23, 24)]),
        ("Aa\U0001f600\U0001f600\u0416-!\u200d\xe9'1Z\u03bbA\u2028\u03bb\u2028", [('Aa', 0, 2), ('\xf0\u0141\u013a\u0122\xf0\u0141\u013a\u0122', 2, 4), ('\xd0\u0138', 4, 5), ('-!\xe2\u0122\u012f', 5, 8), ('\xc3\xa9', 8, 9), ("'", 9, 10), ('1', 10, 11), ('Z\xce\xbbA', 11, 14), ('\xe2\u0122\xa8', 14, 15), ('\xce\xbb', 15, 16), ('\xe2\u0122\xa8', 16, 17)]),
        ("\u0416\u0663'\u2028\x1c\n\u03bb1-\u0301\u0301\u030109\u2163\nb\r\xe9\u06631\u4e2dZA\xa0_9\xa0\xe9\u0416", [('\xd0\u0138', 0, 1), ('\xd9\xa3', 1, 2), ("'", 2, 3), ('\xe2\u0122\xa8', 3, 4), ('\u011c', 4, 5), ('\u010a', 5, 6), ('\xce\xbb', 6, 7), ('1', 7, 8), ('-\xcc\u0123\xcc\u0123\xcc\u0123', 8, 12), ('09\xe2\u0127\xa3', 12, 15), ('\u010a', 15, 16), ('b', 16, 17), ('\u010d', 17, 18), ('\xc3\xa9', 18, 19), ('\xd9\xa31', 19, 21), ('\xe4\xb8\u0143ZA', 21, 24), ('\xc2\u0142', 24, 25), ('_', 25, 26), ('9', 26, 27), ('\xc2\u0142', 27, 28), ('\xc3\xa9\xd0\u0138', 28, 30)]),
        (' 9\n9Z\u2028\t\U0001f600\xb2\U0001f600\u200d\n\u2028\xa0A-\u0416\u0663a\xe9b1', [('\u01209', 0, 2), ('\u010a', 2, 3), ('9', 3, 4), ('Z', 4, 5), ('\xe2\u0122\xa8', 5, 6), ('\u0109', 6, 7), ('\xf0\u0141\u013a\u0122', 7, 8), ('\xc2\xb2', 8, 9), ('\xf0\u0141\u013a\u0122\xe2\u0122\u012f', 9, 11), ('\u010a\xe2\u0122\xa8', 11, 13), ('\xc2\u0142', 13, 14), ('A', 14, 15), ('-', 15, 16), ('\xd0\u0138', 16, 17), ('\xd9\xa3', 17, 18), ('a\xc3\xa9b', 18, 21), ('1', 21, 22)]),
        ("\u0663\u0301 ?a\xa0\u200d\U0001f6001\r\u06631'\U0001f6001\u4e2d", [('\xd9\xa3', 0, 1), ('\xcc\u0123', 1, 2), ('\u0120?', 2, 4), ('a', 4, 5), ('\xc2\u0142', 5, 6), ('\xe2\u0122\u012f\xf0\u0141\u013a\u0122', 6, 8), ('1', 8, 9), ('\u010d', 9, 10), ('\xd9\xa31', 10, 12), ("'\xf0\u0141\u013a\u0122", 12, 14), ('1', 14, 15), ('\xe4\xb8\u0143', 15, 16)]),
        ('Z9?a?-\u2163b\xb2\u200d\x1c\xa0\U0001f600', [('Z', 0, 1), ('9', 1, 2), ('?', 2, 3), ('a', 3, 4), ('?-', 4, 6), ('\xe2\u0127\xa3', 6, 7), ('b', 7, 8), ('\xc2\xb2', 8, 9), ('\xe2\u0122\u012f\u011c', 9, 11), ('\xc2\u0142', 11, 12), ('\xf0\u0141\u013a\u0122', 12, 13)]),
        ("_\u200d'\r\n\xb2\U0001f600?\xb2!?'\u2163!-\xb2", [("_\xe2\u0122\u012f'", 0, 3), ('\u010d', 3, 4), ('\u010a', 4, 5), ('\xc2\xb2', 5, 6), ('\xf0\u0141\u013a\u0122?', 6, 8), ('\xc2\xb2', 8, 9), ("!?'", 9, 12), ('\xe2\u0127\xa3', 12, 13), ('!-', 13, 15), ('\xc2\xb2', 15, 16)]),
        ("_\u4e2d'?!\u0663\u2163\n\xb2", [('_', 0, 1), ('\xe4\xb8\u0143', 1, 2), ("'?!", 2, 5), ('\xd9\xa3\xe2\u0127\xa3', 5, 7), ('\u010a', 7, 8), ('\xc2\xb2', 8, 9)]),
        ('\x1c', [('\u011c', 0, 1)]),
        ('0\u2163\x1c\xe9a\U0001f600\xb2\xb2\n00\r-\U0001f600\u200d\xa0A\u0301!\n\xb2', [('0\xe2\u0127\xa3', 0, 2), ('\u011c', 2, 3), ('\xc3\xa9a', 3, 5), ('\xf0\u0141\u013a\u0122', 5, 6), ('\xc2\xb2\xc2\xb2', 6, 8), ('\u010a', 8, 9), ('00', 9, 11), ('\u010d', 11, 12), ('-\xf0\u0141\u013a\u0122\xe2\u0122\u012f', 12, 15), ('\xc2\u0142', 15, 16), ('A', 16, 17), ('\xcc\u0123!', 17, 19), ('\u010a', 19, 20), ('\xc2\xb2', 20, 21)]),
        ('_\u2028\u0301\n\ra\u2028\U0001f600-', [('_', 0, 1), ('\xe2\u0122\xa8', 1, 2), ('\xcc\u0123', 2, 3), ('\u010a', 3, 4), ('\u010d', 4, 5), ('a', 5, 6), ('\xe2\u0122\xa8', 6, 7), ('\xf0\u0141\u013a\u0122-', 7, 9)]),
        ('\xa0\x1c0\u200d\u03bb?\r_\xa0 \xe9\u03bb\xb2?\U0001f600\u2028\x85\u200d \u0301?\u0301\xb2\x1c?-\xb2\u4e2d--\u2028\t', [('\xc2\u0142', 0, 1), ('\u011c', 1, 2), ('0', 2, 3), ('\xe2\u0122\u012f', 3, 4), ('\xce\xbb', 4, 5), ('?', 5, 6), ('\u010d', 6, 7), ('_', 7, 8), ('\xc2\u0142', 8, 9), ('\u0120\xc3\xa9\xce\xbb', 9, 12), ('\xc2\xb2', 12, 13), ('?\xf0\u0141\u013a\u0122', 13, 15), ('\xe2\u0122\xa8', 15, 16), ('\xc2\u0127', 16, 17), ('\xe2\u0122\u012f', 17, 18), ('\u0120\xcc\u0123?\xcc\u0123', 18, 22), ('\xc2\xb2', 22, 23), ('\u011c?-', 23, 26), ('\xc2\xb2', 26, 27), ('\xe4\xb8\u0143', 27, 28), ('--', 28, 30), ('\xe2\u0122\xa8\u0109', 30, 32)]),
        ('\t\U0001f600\xe9\xa0\xa0!\u0663\u0416 \u0301\n\n\n0\u20280b\xb2\u2028\tZ\u0663\u2163b1', [('\u0109', 0, 1), ('\xf0\u0141\u013a\u0122', 1, 2), ('\xc3\xa9', 2, 3), ('\xc2\u0142', 3, 4), ('\xc2\u0142', 4, 5), ('!', 5, 6), ('\xd9\xa3', 6, 7), ('\xd0\u0138', 7, 8), ('\u0120\xcc\u0123', 8, 10), ('\u010a\u010a', 10, 12), ('\u010a', 12, 13), ('0', 13, 14), ('\xe2\u0122\xa8', 14, 15), ('0', 15, 16), ('b', 16, 17), ('\xc2\xb2', 17, 18), ('\xe2\u0122\xa8', 18, 19), ('\u0109', 19, 20), ('Z', 20, 21), ('\xd9\xa3\xe2\u0127\xa3', 21, 23), ('b', 23, 24), ('1', 24, 25)]),
        ('\u03bb\xe9 b-\u04169\u0416\r\xb2\x1c\t\u2163\u03bb\x85-\U0001f600_\u2163\xb2\x1c\x1c\u4e2d\xe9', [('\xce\xbb\xc3\xa9', 0, 2), ('\u0120b', 2, 4), ('-', 4, 5), ('\xd0\u0138', 5, 6), ('9', 6, 7), ('\xd0\u0138', 7, 8), ('\u010d', 8, 9), ('\xc2\xb2', 9, 10), ('\u011c', 10, 11), ('\u0109', 11, 12), ('\xe2\u0127\xa3', 12, 13), ('\xce\xbb', 13, 14), ('\xc2\u0127', 14, 15), ('-\xf0\u0141\u013a\u0122_', 15, 18), ('\xe2\u0127\xa3\xc2\xb2', 18, 20), ('\u011c\u011c', 20, 22), ('\xe4\xb8\u0143\xc3\xa9', 22, 24)]),
        ('\u0416_b\u200db\xa0\n9_\x850\u4e2d\u2028\x1cA\u03bb\u0663?', [('\xd0\u0138', 0, 1), ('_', 1, 2), ('b', 2, 3), ('\xe2\u0122\u012f', 3, 4), ('b', 4, 5), ('\xc2\u0142', 5, 6), ('\u010a', 6, 7), ('9', 7, 8), ('_', 8, 9), ('\xc2\u0127', 9, 10), ('0', 10, 11), ('\xe4\xb8\u0143', 11, 12), ('\xe2\u0122\xa8', 12, 13), ('\u011c', 13, 14), ('A\xce\xbb', 14, 16), ('\xd9\xa3', 16, 17), ('?', 17, 18)]),
    ],
}


def _assert_result(text, actual, expected=None):
    assert isinstance(actual, list), 'Return a list'
    cursor = 0
    decoded = []
    for row in actual:
        assert isinstance(row, tuple) and len(row) == 3, row
        piece, start, end = row
        assert isinstance(piece, str) and piece, row
        assert type(start) is int and type(end) is int, row
        assert start == cursor and start < end <= len(text), row
        raw = bytes(_INVERSE[char] for char in piece)
        assert raw.decode('utf-8') == text[start:end], (text, row)
        decoded.append(raw)
        cursor = end
    assert cursor == len(text)
    assert b''.join(decoded) == text.encode('utf-8')
    if expected is not None:
        assert actual == expected, (repr(text), actual, expected)


def _group(solve, name):
    for text, expected in _FIXTURES[name]:
        _assert_result(text, solve(text), expected)


def check_empty_and_single(solve):
    _group(solve, 'empty')


def check_lowercase_contractions(solve):
    _group(solve, 'contractions')


def check_case_sensitive_contractions(solve):
    _group(solve, 'case')


def check_punctuation_priority(solve):
    _group(solve, 'punctuation')


def check_unicode_letters_no_normalization(solve):
    _group(solve, 'letters')


def check_all_numeric_categories(solve):
    _group(solve, 'numbers')


def check_combining_marks_and_emoji(solve):
    _group(solve, 'marks_emoji')


def check_internal_and_trailing_spaces(solve):
    _group(solve, 'spaces')


def check_explicit_unicode_whitespace(solve):
    _group(solve, 'whitespace')
    whitespace = '\t\n\v\f\r \u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000'
    for char in whitespace:
        encoded = ''.join(_BYTE_ALPHABET[b] for b in char.encode('utf-8'))
        expected = [('a', 0, 1), (encoded, 1, 2)]
        expected += [(encoded + 'b', 2, 4)] if char == ' ' else [(encoded, 2, 3), ('b', 3, 4)]
        _assert_result('a' + char * 2 + 'b', solve('a' + char * 2 + 'b'), expected)


def check_controls_are_not_isspace(solve):
    _group(solve, 'controls')


def check_original_codepoint_offsets(solve):
    _group(solve, 'offsets')


def check_frozen_seeded_random(solve):
    assert len(_FIXTURES['random']) == 24
    _group(solve, 'random')


def check_maximum_length(solve):
    for text, piece in [('a' * 300, 'a' * 300), ('é' * 300, 'Ã©' * 300), (' ' * 300, 'Ġ' * 300)]:
        _assert_result(text, solve(text), [(piece, 0, 300)])
    text = 'a1' * 150
    expected = [(char, i, i + 1) for i, char in enumerate(text)]
    _assert_result(text, solve(text), expected)


def check_full_byte_table_and_reachable_utf8(solve):
    assert len(_BYTE_ALPHABET) == len(_INVERSE) == 256
    assert bytes(_INVERSE[char] for char in _BYTE_ALPHABET) == bytes(range(256))
    # Generated by greedy byte coverage over assigned Unicode 15.0 scalars.
    # 240 characters cover all 241 byte values reachable under this contract.
    text = ''.join(map(chr, range(192))) + (
        '\xc0\u0100\u0140\u0180\u01c0\u0200\u0240\u0280\u02c0\u0300\u0340\u0384\u03c0'
        '\u0400\u0440\u0480\u04c0\u0500\u0540\u0580\u05c0\u0600\u0640\u0680\u06c0'
        '\u0700\u0740\u0780\u07c0\u0800\u1000\u2000\u3000\u4000\u5000\u6000\u7000'
        '\u8000\u9000\ua000\ub000\uc000\ud000\ue000\uf000\U00010000\U000e0001\U00100000'
    )
    unreachable = {0xC0, 0xC1, 0xF1, 0xF2, *range(0xF5, 0x100)}
    assert len(text) == 240
    assert set(text.encode('utf-8')) == set(range(256)) - unreachable
    _assert_result(text, solve(text))
    # Exercise every reachable mapping independently of the scanner oracle.
    for char in text:
        piece = ''.join(_BYTE_ALPHABET[b] for b in char.encode('utf-8'))
        _assert_result(char, solve(char), [(piece, 0, 1)])


check_empty_and_single.example = True
check_lowercase_contractions.example = True
check_punctuation_priority.example = True
check_original_codepoint_offsets.example = True
