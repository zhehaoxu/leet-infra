# Chinese TN number verbalization rule engine

## Background and sources

The TTS text frontend (TN / text normalization) turns written text into a speakable
string. PaddleSpeech `paddlespeech/t2s/frontend/zh_normalization/` (`num.py`,
`chronology.py`, `phonecode.py`; rules ported from WeTextProcessing) contains a purely
rule-based number verbalizer that transcribes fully. This problem pins every rule to
the official source.

## Mechanism ledger

1. **Myriad recursion** (`_get_value`): with `stripped = value.lstrip('0')`, pick the
   largest power in `UNITS = {1:十, 2:百, 3:千, 4:万, 8:亿}` with `power < len(stripped)`
   and split `value[:-k] / value[-k:]` **on the original string** (not stripped), then
   recurse. The unit symbol (万/亿) sits between the halves.
2. **Mid-string zero padding**: at recursion leaves, stripped of length 1 but shorter
   than its input (e.g. the "1" of "001") -> `[零, X]`; empty stripped -> `[]`. Leading
   zeros die at the initial lstrip ("07" -> "七"), while trailing-segment zeros are
   padded ("1001" -> "一千零一") — the two zero behaviors are asymmetric.
3. **"一十" abbreviation**: applies only to the **whole symbol string's** head
   ("10"->"十", "100000"->"十万", "1000000010"->"十亿零一十"); inside segments no
   abbreviation ("110" -> "一百一十").
4. **num2str decimals**: decimal part `rstrip('0')` ("3.20"->"三点二"); empty integer
   reading becomes "零" (".22"->"零点二二"); decimal digits read one by one, zeros
   included ("0.05"->"零点零五").
5. **Time**: minute == 30 special-cased to "半"; minute/second go through num2str after
   lstrip, with a "零" prepended when the original started with 0 ("05"->"零五");
   all-zero fields omitted ("8:00"->"八点").
6. **Phone numbers**: per-digit reading with alt_one (`replace('一','幺')` affects only
   digit 1), mobile splits on spaces (after stripping a leading '+'), fixed lines split
   on "-", segments joined by full-width "，".

## Independent oracle and cross-checks

- Fixed anchors cover all quirks; on 300 random integers the official string recursion
  and an **independent numeric method** (myriad 4-digit sectioning + in-section zero
  compression + inter-section zero padding + head-only 一十 abbreviation) agree.
- Another 200 random cross-checks for decimals/times/phones.

## Mutations

14/14 caught: dropping lstrip, all-zero -> empty string, power condition `<`->`<=`,
dropping mid-zero padding, loosening the 一十 abbreviation, keeping decimal trailing
zeros, no zero for empty integer, dropping the 半 special case, dropping the 零五
padding, alt_one always on / wrong replace string, phone without strip('+'), fixed
splitting on spaces, full-width comma -> ASCII.
