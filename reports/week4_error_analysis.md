# Week 4 error analysis

Generated from `scripts/run_week4_eval.py` predictions in `reports/week4_eval/predictions`. Repetition-loop flagging is a cheap n-gram heuristic (see `scripts/run_week4_eval.py::repetition_flag`) — flagged rows still need a human read, they are candidates, not confirmed failures.

## en-sw  (n=304)

### Per-domain scores

| Domain | n | BLEU | chrF | Repetition-flagged |
|---|---:|---:|---:|---:|
| Agriculture | 19 | 61.73 | 78.90 | 0 |
| Education | 41 | 52.85 | 76.50 | 0 |
| Governance | 23 | 59.57 | 79.96 | 0 |
| Health | 177 | 44.76 | 68.96 | 0 |
| Security | 44 | 57.41 | 77.73 | 0 |

### 8 lowest-chrF examples

| Domain | chrF | Source | Reference | Model output |
|---|---:|---|---|---|
| Health | 29.9 | Bilateral multilobar ground-glass opacities with a peripheral, asymmetric and posterior distribution are common in early | Glasi ya laini mbili na vyuma vingi yenye sehemu za pembeni inayotumika kuchunguza kamba za pumu , | Ukosefu wa uwazi wa kioo cha chini cha vitambaa vingi vya pande mbili na usambazaji wa nje, usio sawa na wa nyuma ni kaw |
| Health | 33.9 | Allergic contact dermatitis, contact urticaria syndrome or hypersensitivity to alcohol or additives present in alcohol h | Matatizo ya mzio unaotakana na mguso wa ngozi, ugonjwa wa kuchubuka kwa ngozi au matatizo yanayotokana na alkoholi au vi | Uvimbe wa ngozi wa kugusana na kingamwili, ugonjwa wa kugusana na urticaria au unyeti kupita kiasi wa pombe au viungo vi |
| Health | 38.3 | Thus far, WIV1 represents the most closely related ancestor of SARS-CoV in bats, sharing 95% nucleotide sequence homolog | Kufikia sasa, virusi vya WIV1 vinawakilisha kizazi cha karibu sana cha SARS-CoV kwenye popo, vikiwa na 95% za msururu wa | Hadi sasa, WIV1 inawakilisha mzazi wa karibu zaidi wa SARS-CoV katika popo, akishiriki 95% ya homolojia ya mfuatano wa n |
| Health | 41.9 | An animal serves as the evolutionary host of an HCoV if it harbours a closely related ancestor sharing high homology at  | Mnyama hutumika kama kibebaji cha HCoV cha kiukuaji ikiwa amehifadhi ukoo wa karibu unaofanana katika kiwango cha mfulul | Mnyama hutumika kama mwenyeji wa ukuaji wa HCoV ikiwa ana mzazi wa karibu anayeshiriki homolojia ya juu katika ngazi ya  |
| Education | 42.2 | Clubs to perform for civic week | Vilabu vyatumbuiza kwa wiki ya uraia | Vilabu vitatoa tamasha kwa ajili ya wiki ya kiraia |
| Health | 42.6 | But it's almost impossible to make any sensible projection right now about when that will be". | Lakini ni vigumu kukadiria au kuwa na makadirio elekevu kwa sasa kuhusu wakati huo ni lini". | Lakini ni karibu haiwezekani kufanya utabiri wowote unaofaa sasa kuhusu wakati huo utakapokuwa". |
| Health | 43.8 | We would not be working in isolation on this research. | Hatutafanya kazi pekee yetu katika utafiti huu. | Hatutakuwa tukifanya utafiti huu kwa njia ya kujitenga. |
| Health | 43.9 | For the diagnosis of “suspected case” and “confirmed case”, we suggest to trace and obey the newest guidelines of their  | Kwa upimaji wa "kisa kinachoshukiwa" na "kisa kilichothibitishwa", tunapendekeza kufuatwa na utiifu wa miongozo mipya ya | Kwa ajili ya utambuzi wa "kosa linalotiliwa shaka" na "kosa lililothibitishwa", tunashauri kufuatilia na kutii miongozo  |

### Repetition-loop flagged examples (up to 8)

_None flagged in this direction._

## sw-en  (n=304)

### Per-domain scores

| Domain | n | BLEU | chrF | Repetition-flagged |
|---|---:|---:|---:|---:|
| Agriculture | 19 | 50.52 | 72.47 | 0 |
| Education | 41 | 53.71 | 72.69 | 0 |
| Governance | 23 | 54.18 | 74.27 | 0 |
| Health | 177 | 44.39 | 65.61 | 0 |
| Security | 44 | 56.71 | 76.56 | 0 |

### 8 lowest-chrF examples

| Domain | chrF | Source | Reference | Model output |
|---|---:|---|---|---|
| Health | 21.9 | Hatutafanya kazi pekee yetu katika utafiti huu. | We would not be working in isolation on this research. | We won't do our work alone in this study. |
| Health | 22.0 | Glasi ya laini mbili na vyuma vingi yenye sehemu za pembeni inayotumika kuchunguza kamba za pumu , | Bilateral multilobar ground-glass opacities with a peripheral, asymmetric and posterior distribution are common in early | Two-ply, multi-metal glass with angular sections used to examine respiratory strings , |
| Health | 22.6 | Taiwan iliongeza utengenezaji wa barakoa za uso na kutoza faini kwa kuhodhi bidhaa za kimatibabu. | Taiwan increased face mask production and penalized hoarding of medical supplies.Simulations for Great Britain and the U | Taiwan increased manufacturing of face masks and imposed fines for ordering medical supplies. |
| Health | 28.7 | Matatizo ya mzio unaotakana na mguso wa ngozi, ugonjwa wa kuchubuka kwa ngozi au matatizo yanayotokana na alkoholi au vi | Allergic contact dermatitis, contact urticaria syndrome or hypersensitivity to alcohol or additives present in alcohol h | Skin-to-touch allergy problems, dermatitis or alcohol-induced problems or alcohol-containing organs were less common. |
| Health | 30.3 | Epuka kukumbatiana, kupigana busu, kusalimiana kwa mikono, kusalimiana kwa ngumi, na mgusano wa aina yoyote. | Avoid hugs, kisses, handshakes, fist bumps, and any other contact. | Avoid hugging, kissing, shaking hands, punching, and touching of any kind. |
| Health | 33.7 | Kwa vile tunaweka muundo wa janga wa SIR katika utarbitbu wa ABC, tunatarajia kwamba matokeo yetu yatakuwa makubwa yakil | Since we are fitting an SIR-epidemic model in the ABC routine, we anticipate that our results will be robust against wee | As we place the SIR pandemic pattern in ABC processing, we expect that our results will be large compared to weekly data |
| Education | 33.7 | Walimu wakuu lazima wabandike orodha ya wafaidi wa misaada ya masomo kwenye ubao wa matangazo wa shule kwa uwazi. | Head teachers must display the list of bursary beneficiaries on the school noticeboard for transparency. | Principals must clearly list scholarship beneficiaries on school billboards. |
| Health | 33.9 | wageni wasio wakazi wamekatazwa kuingia ghorofa za kuishi | non-resident visitors prohibited from entering apartment complexes | non-resident foreigners banned from living quarters |

### Repetition-loop flagged examples (up to 8)

_None flagged in this direction._

## en-guz  (n=138)

### Per-domain scores

| Domain | n | BLEU | chrF | Repetition-flagged |
|---|---:|---:|---:|---:|
| Agriculture | 19 | 4.03 | 28.34 | 3 |
| Education | 38 | 3.91 | 29.30 | 3 |
| Governance | 18 | 6.42 | 34.16 | 1 |
| Health | 25 | 3.07 | 24.30 | 8 |
| Security | 38 | 2.01 | 26.15 | 8 |

### 8 lowest-chrF examples

| Domain | chrF | Source | Reference | Model output |
|---|---:|---|---|---|
| Health | 6.3 | Take a journal and note down how your days are and how you're feeling. Examine your thoughts written over time and soon  | Bwata egetabu,rika inaki amatuko ao are, nakomenta buna oigwete.
 Erigererie na gotuka ebirengererio biago biria kwarige | Karwe ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero  |
| Security | 6.9 | Encrypt sensitive files before sharing them online. | Amanagana ao obobisi yakeborane gochia enkwana yobobisi otayarasanga ase eintaneti. | Bwatia omoroberio bwobotuki bwobotuki bwobotuki bwobotuki bwobotuki bwobotuki bwobotuki bwobotuki bwobotuki bwobotuki bw |
| Security | 9.6 | Learn to build temporary shelters using locally available materials for displaced families. | Egera koacha chinyomba ogtumia ebinto bigotoka inka ase bari baseretigwe korwa aase bare abwo nyuma. | Egera buna ogochaka chinsemo chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria  |
| Governance | 9.9 | KRA ON THE ROAD Tax man tours Mombasa County, residents urged to file their returns | EKEOMBE KIOGOSANGERERIA EBANGO. Etaro yomonachi ebango omonene ime yekaunti yamombasa abamenyi baborigwe koirania amang' | KRA ON THE ROAD Tax man tours Mombasa County, residents urged to file their returns |
| Health | 10.5 | Keep hot drinks and cooking pots away from childrens reach to prevent burn injuries. | Beka are ebinyugwa bire morero na chisuguria chiokorugerwa korwa ase abana gotanga emebasokano | Renda ebinto biria bigotumia ebinto biria bigotumia ebinto biria bigotumia ebinto biria bigotumia ebinto biria bigotumia |
| Security | 10.5 | Terror threats remain high; avoid high‑profile crowded areas. | Obwoba bogakwa chipomu nigo bore igoro; erende toba ase omosangerekano bwa abanto abange. | Oboremi bwokoruta chipomu bwokoruta chipomu bwokoruta chipomu bwokoruta chipomu chikoruta chipomu chikoruta chipomu chik |
| Security | 10.6 | Don't display valuables in your car - keep them out of sight. | Tiga kworokia ebinto bierigori ime yerori yao - keep them outside. | Tobaise gokora ebinto biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria bi |
| Health | 10.6 | Sleeping under treated nets reduces malaria risk for pregnant women and unborn babies. | Korara ime yechineti chibekire amariogo ngokeani chire ekeugoso kia esosera ase abàngina bebwateraneti na abana bataraib | Korara ase chinsemo chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria ch |

### Repetition-loop flagged examples (up to 8)

| Domain | Model output (truncated) |
|---|---|
| Education | Omoroberio bweserikari ase abana bonsi ba TVET baria bakorenta omotienyi omotambe omonene omotambe omotambe omotambe omotambe omotambe omotambe omotambe omotamb |
| Education | Omonto omonyete goikerania chisukuru ase chibesa chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria |
| Education | Kabe maiso! Bwatia omoroberio bwogotumia ebinto biria bigotumia ebinto biria bigotumia ebinto biria bigotumia ebinto biria bigotumia ebinto biria bigotumia ebin |
| Health | Kenya yanyorire ebinyoro biabanto bonsi, chigosori, ebinyoro, na chigosori chieserekari; chigosori chiria chiria chiria chiria chiria chiria chiria chiria chiri |
| Health | Ekeombe keria ekenene getenerete amagenderero (WHO) nkiagendererete omoroberio o'Mpox, nkiagera ng'a omoroberio o'Mpox nogokorera oborwaria bwobobwenia bwoborwa |
| Health | Karwe ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero eker |
| Health | Ekeombe keria ekenene getenenerete oborwaire bwa Esosera ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero ekero e |
| Health | Tumia amache abwatirwe gose amache abwatirwe gosegeta chindagera chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chiria chir |

## guz-en  (n=138)

### Per-domain scores

| Domain | n | BLEU | chrF | Repetition-flagged |
|---|---:|---:|---:|---:|
| Agriculture | 19 | 2.90 | 17.53 | 0 |
| Education | 38 | 3.16 | 19.16 | 0 |
| Governance | 18 | 4.31 | 19.45 | 0 |
| Health | 25 | 5.43 | 22.33 | 0 |
| Security | 38 | 1.00 | 18.86 | 0 |

### 8 lowest-chrF examples

| Domain | chrF | Source | Reference | Model output |
|---|---:|---|---|---|
| Agriculture | 5.2 | Omoroberio oyo nomoikerabu, omoroberio omotambe omotwe ase ogotumia amoyo yeubi gokora obwoereria bwenchindagera, orenge | The strategy is a holistic, long-term approach using agroecology to address food system challenges, aiming for food secu | The omoroberio or omoroberio or omoroberio or omoroberio or omoroberio or omoroberio or omoroberio or omoroberio or omor |
| Education | 8.3 | Ogosomia goetera egedichitari bosa ochakirwe. | Free national revision TV channel launches Dec 1 – KBC Channel 23, 24/7! | Ogosomia goetera egedichitari bosa ochakirwe. |
| Education | 10.9 | Egekwegia buna ense ekorigia, kobanga na gotumia chibesa. | Simulates real-life county fiscal choices | Egekwegia buna ense ekorigia, kobanga and gotumia chibesa. |
| Education | 11.0 | Sira chisemi! Ebinyoro, konya korwa esukuru fisi. | Support education! Communities, fund school fees. | Sira chisimi! Ebinyinyi, konya korwa esukuru fisi. |
| Education | 11.2 | Toma amang'ana ameng'e goetera esimi arengete obogesaku. | SMS-based quizzes with civic themes | Toma touches me and goetera esimi and obogesaku. |
| Education | 11.2 | Keania ogotareng'ana. Karwe ebigokonya abana goikera etkinorochia. | Bridge the gap! Provide digital tools for disabled students. | Keania ogotareng'ana. Karwe ebigokonya abana goikera etkinorochia. |
| Education | 11.9 | Kae chisukuru chibesa! Rora ng'a chibesa chiria eserikari ekorwa ase kera omwana chiakeranire gokonya emeroberio yesukur | Fund schools! Ensure timely capitation for stability. | Kae chisukuru chibesa! Rora ng'a chibesa chiria eserikari ekorwa ase kera omwana chiakeranire gokonya emeroberio yesukur |
| Education | 11.9 | Ribaga riabeene ase abana baria bare nokogania kwabeene. | Priority for bursary to needy students. | Ribaga ribeene ase abana bare bare nokogania gointo each other. |

### Repetition-loop flagged examples (up to 8)

_None flagged in this direction._
