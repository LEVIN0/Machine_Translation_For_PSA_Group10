# Team-Written PSA Kit (Week 2 Topic Expansion)

Goal: each member writes authentic-style Kenyan PSA sentences to expand topic
coverage beyond the scraped sources, then adds a Kiswahili translation (and
Ekegusii, if a speaker is available). These are **original works** — do not
copy text from any website.

**How to submit:** copy `data/manual/team_psas_template.csv` to
`data/manual/<your_name>.csv`, fill in your rows (one sentence per row), and
commit. Files named `*_template.csv` are examples and are skipped by the
importer; every other `*.csv` in `data/manual/` is imported by
`src/corpora/manual.py` during `build()`.

## Style rules

1. **One action per sentence.** If the sentence has two instructions, split it.
2. **Directive voice**: start with a verb ("Visit", "Report", "Register",
   "Vaccinate") or a clear public notice ("The County Government announces…").
3. **Present tense**, 6–20 words, plain vocabulary (primary-school reading
   level).
4. **Kenyan-institution voice**: refer to real structures — Ministry of
   Health, County Government, area chief, Huduma Centre, NTSA, chiefs' barazas.
5. Use Kenyan cultural/institutional terms naturally (see
   `data/glossary.json`) — matatu, boda boda, harambee, M-Pesa, NHIF/SHA.
6. English first; the **Kiswahili column is filled by the team** (native
   speakers review). Ekegusii only if a speaker is available.
7. Every row must carry one of the 25 sub-topics below in `Notes`.

## The 25 sub-topics (5 domains × 5) with worked examples

### Health

1. **Disease prevention**
   > Wash your hands with soap and clean water before every meal.
2. **Maternal & child health**
   > Take your child to the nearest clinic for all scheduled immunisations.
3. **Public health campaigns**
   > Attend the free cholera vaccination campaign at your nearest health centre this week.
4. **Mental health**
   > Seek help at the county hospital if you feel overwhelmed; mental health is health.
5. **Healthcare access**
   > Register with the Social Health Authority today to access affordable hospital care.

### Agriculture

6. **Crop production**
   > Plant certified drought-resistant maize seed before the long rains begin.
7. **Livestock**
   > Vaccinate your cattle against foot-and-mouth disease at the ward veterinary camp.
8. **Agribusiness & market access**
   > Join your local farmers' sacco to sell your harvest at fair market prices.
9. **Sustainable farming**
   > Plant cover crops on your shamba to protect the soil during the dry season.
10. **Agricultural training**
    > Register for the free extension training at your sub-county agriculture office.

### Education

11. **Access to education**
    > Enrol every school-age child in the nearest public school; education is free.
12. **Vocational training**
    > Apply for a TVET course at your nearest vocational training centre this month.
13. **Civic education**
    > Attend the civic education baraza at the chief's camp on Saturday morning.
14. **Educational resources**
    > Collect free revision books for KCSE candidates from your ward library.
15. **School safety & inclusion**
    > Report any case of bullying or discrimination to your school head teacher.

### Security

16. **Public safety awareness**
    > Slow down and obey traffic rules; road safety begins with you.
17. **Crime prevention**
    > Join your nyumba kumi group and report crime to the area chief or police.
18. **National security**
    > Report any suspicious person or activity to the National Police Service hotline.
19. **Gender-based violence**
    > Speak out against gender-based violence and report cases to the nearest police station.
20. **Cybersecurity**
    > Never share your M-Pesa PIN or one-time passwords with anyone, even family.

### Governance

21. **Anti-corruption**
    > Refuse to pay bribes and report corruption to the EACC anonymously.
22. **Public participation**
    > Attend the county budget public participation forum in your ward this week.
23. **Elections & voter education**
    > Check your voter registration details at the nearest IEBC office today.
24. **Public service delivery**
    > Apply for your national ID and birth certificate at any Huduma Centre near you.
25. **Devolution & local governance**
    > Pay your county business permit on time to keep local services running.

## After writing

- Add the Kiswahili column (team review: one native/fluent speaker checks all).
- Keep the `Notes` column: sub-topic + author initials.
- Run `python -c "from src.corpora.manual import import_manual; import_manual()"`
  to confirm your rows import cleanly before committing.
