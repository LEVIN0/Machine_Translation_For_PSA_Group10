# Week 4 error analysis

Generated from `scripts/run_week4_eval.py` predictions in `reports/week4_eval_aug/predictions`. Repetition-loop flagging is a cheap n-gram heuristic (see `scripts/run_week4_eval.py::repetition_flag`) — flagged rows still need a human read, they are candidates, not confirmed failures.

## en-sw  (n=304)

### Per-domain scores

| Domain | n | BLEU | chrF | Repetition-flagged |
|---|---:|---:|---:|---:|
| Agriculture | 19 | 58.81 | 78.77 | 0 |
| Education | 41 | 52.25 | 75.86 | 0 |
| Governance | 23 | 58.83 | 78.76 | 0 |
| Health | 177 | 45.02 | 70.02 | 0 |
| Security | 44 | 56.39 | 77.35 | 0 |

### 8 lowest-chrF examples

| Domain | chrF | Source | Reference | Model output |
|---|---:|---|---|---|
| Health | 32.8 | Allergic contact dermatitis, contact urticaria syndrome or hypersensitivity to alcohol or additives present in alcohol h | Matatizo ya mzio unaotakana na mguso wa ngozi, ugonjwa wa kuchubuka kwa ngozi au matatizo yanayotokana na alkoholi au vi | Ugonjwa wa ngozi wa kugusana na kingamwili, ugonjwa wa kugusana na urticaria au unyeti kupita kiasi kwa pombe au virutub |
| Health | 34.7 | Bilateral multilobar ground-glass opacities with a peripheral, asymmetric and posterior distribution are common in early | Glasi ya laini mbili na vyuma vingi yenye sehemu za pembeni inayotumika kuchunguza kamba za pumu , | Ukosefu wa uwazi wa kioo cha chini chenye sehemu nyingi na usambazaji wa pembeni, usio sawa na wa nyuma ni wa kawaida ka |
| Health | 41.2 | But it's almost impossible to make any sensible projection right now about when that will be". | Lakini ni vigumu kukadiria au kuwa na makadirio elekevu kwa sasa kuhusu wakati huo ni lini". | Lakini ni karibu haiwezekani kufanya utabiri wowote unaofaa sasa kuhusu wakati huo utakapotokea". |
| Health | 42.2 | Particularly, bat CoVs with zoonotic potential are so diverse. | Hususan, virusi vya CoV vinavyosababishwa na popo vyenye uwezo wa kusambazwa na wanyama ni anuwai mno. | Hasa, virusi vya CoV vya popo vinavyoweza kuambukiza wanyama ni tofauti sana. |
| Education | 42.2 | Clubs to perform for civic week | Vilabu vyatumbuiza kwa wiki ya uraia | Vilabu vitatoa tamasha kwa ajili ya wiki ya kiraia |
| Education | 45.2 | SMS-based quizzes with civic themes | Majaribio yanayotegemea SMS yenye mandhari ya kiraia | Maswali yanayotokana na SMS yenye mada za kiraia |
| Education | 45.4 | Simulates real-life county fiscal choices | Huiga chaguzi halisi za fedha za kaunti | Inasimulia uchaguzi wa kifedha wa maisha halisi wa kaunti |
| Security | 46.1 | Nairobi informal settlements community policing case study with relevance to wider Nakuru experience. | Uchunguzi wa kesi za polisi jamii katika makazi yasiyo rasmi ya Nairobi unaohusiana na uzoefu mpana wa Nakuru. | Nairobi inaanzisha utafiti wa kesi ya polisi wa jamii na umuhimu kwa uzoefu mpana wa Nakuru. |

### Repetition-loop flagged examples (up to 8)

_None flagged in this direction._

## sw-en  (n=304)

### Per-domain scores

| Domain | n | BLEU | chrF | Repetition-flagged |
|---|---:|---:|---:|---:|
| Agriculture | 19 | 49.13 | 70.82 | 0 |
| Education | 41 | 53.65 | 73.29 | 0 |
| Governance | 23 | 52.96 | 73.23 | 0 |
| Health | 177 | 46.03 | 66.80 | 0 |
| Security | 44 | 54.61 | 75.36 | 0 |

### 8 lowest-chrF examples

| Domain | chrF | Source | Reference | Model output |
|---|---:|---|---|---|
| Health | 16.9 | Glasi ya laini mbili na vyuma vingi yenye sehemu za pembeni inayotumika kuchunguza kamba za pumu , | Bilateral multilobar ground-glass opacities with a peripheral, asymmetric and posterior distribution are common in early | Double-sided and multi-metal angular glass used to scan respiratory tractors , |
| Health | 22.6 | Taiwan iliongeza utengenezaji wa barakoa za uso na kutoza faini kwa kuhodhi bidhaa za kimatibabu. | Taiwan increased face mask production and penalized hoarding of medical supplies.Simulations for Great Britain and the U | Taiwan increased manufacturing of face masks and imposed fines for ordering medical supplies. |
| Health | 29.0 | Hususan, virusi vya CoV vinavyosababishwa na popo vyenye uwezo wa kusambazwa na wanyama ni anuwai mno. | Particularly, bat CoVs with zoonotic potential are so diverse. | Specifically, bat-caused CoVs capable of transmission to animals are very diverse. |
| Education | 29.4 | Vilabu vyatumbuiza kwa wiki ya uraia | Clubs to perform for civic week | Clubs entertain for citizenship week |
| Health | 30.3 | Kwa vile tunaweka muundo wa janga wa SIR katika utarbitbu wa ABC, tunatarajia kwamba matokeo yetu yatakuwa makubwa yakil | Since we are fitting an SIR-epidemic model in the ABC routine, we anticipate that our results will be robust against wee | As we place the epidemic pattern of SIR in ABC processing, we expect that our findings will be large compared to weekly  |
| Health | 33.2 | wageni wasio wakazi wamekatazwa kuingia ghorofa za kuishi | non-resident visitors prohibited from entering apartment complexes | non-resident aliens banned from living quarters |
| Health | 34.0 | Matatizo ya mzio unaotakana na mguso wa ngozi, ugonjwa wa kuchubuka kwa ngozi au matatizo yanayotokana na alkoholi au vi | Allergic contact dermatitis, contact urticaria syndrome or hypersensitivity to alcohol or additives present in alcohol h | Skin contact-resistant allergies, dermatitis or alcohol-induced disorders or alcohol-containing organs are less common. |
| Education | 40.6 | Walimu wakuu lazima wabandike orodha ya wafaidi wa misaada ya masomo kwenye ubao wa matangazo wa shule kwa uwazi. | Head teachers must display the list of bursary beneficiaries on the school noticeboard for transparency. | Principals must transparently list scholarship beneficiaries on school billboards. |

### Repetition-loop flagged examples (up to 8)

_None flagged in this direction._

## en-guz  (n=138)

### Per-domain scores

| Domain | n | BLEU | chrF | Repetition-flagged |
|---|---:|---:|---:|---:|
| Agriculture | 19 | 4.17 | 30.52 | 3 |
| Education | 38 | 6.20 | 33.48 | 2 |
| Governance | 18 | 5.74 | 32.45 | 1 |
| Health | 25 | 4.37 | 31.59 | 2 |
| Security | 38 | 2.92 | 29.04 | 6 |

### 8 lowest-chrF examples

| Domain | chrF | Source | Reference | Model output |
|---|---:|---|---|---|
| Governance | 1.1 | INCREASING FOREST COVER 14,000 trees planted in Kitui Residents urged to continue planting trees | ASE OGOKINIA AMANANI EMETE 14, 000  gosimekwa ime ya kitui, nabamenyi koborigwa gokong'a ase okogenderera gosimeka emete | OBONO BWA BONGO BONGO BONGO BONGO BONGO BONGO BONGO BONGO BONGO BONGO BONGO BONGO BONGO BONGO BONGO BONGO BONGO BONGO BO |
| Education | 6.4 | All Form Four students must register as voters before leaving school – IEBC directive. | Abana besekondari bekerasi kia inye goika babe berikire koba abbaki chikura batrarua esukuru | Abana bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi  |
| Security | 6.6 | Hilly areas like Murang'a are prone to landslides. Avoid building on steep slopes and heed evacuation warnings during he | Chinsemo chimogete iguru buna Murang'a naororo konyora okwanyoka kweriroba.Kabe are nechinyomba chiagachire ase aimogete | Ebinyoro biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria bir |
| Education | 7.2 | Free national revision TV channel launches Dec 1 – KBC Channel 23, 24/7! | Ogosomia goetera egedichitari bosa ochakirwe. | Ekeombe keria ekenene getenerete obokoreri ase omoroberio bwokorobererekana ase omoroberio bwokorobererekana ase omotien |
| Health | 12.3 | Today, we have recorded 447 new positive cases out of 3803 samples bringing the total of positive cases in the country t | Rero, twanyorire abarwaire 447 ase abanto 3803 banyarire gopimwa, bonsi banyarire goikia abarwaire 8,975. Chisampuli chi | Ekeombe keria ekenene getenenerete obokoreri bwechinyomba chiokwegendereria chiokwegendereria chiokwegendereria chiokweg |
| Health | 12.8 | Take a journal and note down how your days are and how you're feeling. Examine your thoughts written over time and soon  | Bwata egetabu,rika inaki amatuko ao are, nakomenta buna oigwete.
 Erigererie na gotuka ebirengererio biago biria kwarige | Beka ekeombe keria ekiagera na gosaba ng'a ebiro bionsi bionsi bionsi bionsi bionsi bionsi bionsi bionsi bionsi bionsi b |
| Health | 12.9 | Children exposed to smoke are more likely to develop asthma and respiratory infections. | Abana bakongusa amarioki nkonyora bare amarwaire ya amayaa na okoeyana | Abana baria bakorwe ase omoroberio bwogotumia omoroberio bwogotumia omoroberio bwokoruta omoroberio bwokoruta omoroberio |
| Agriculture | 13.1 | Push-pull technology is an approach which involves using repellant plants (silver leaf desmodium) to push pests away fro | Ogosegeta omoroberio otekinorochia nenchera ierengete goyumia ebimeri biria bigoseria ebimeria biri bende bikorenta emec | Teknolojia ya kusukuma na kusukuma ni mbinu inayohusisha kutumia mimea inayosambaza (desmodium ya majani ya fedha) kusuk |

### Repetition-loop flagged examples (up to 8)

| Domain | Model output (truncated) |
|---|---|
| Education | Abana bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bonsi bons |
| Education | Ekeombe keria ekenene getenerete obokoreri ase omoroberio bwokorobererekana ase omoroberio bwokorobererekana ase omotienyi omotangani 1 omotienyi omotangani. Ek |
| Health | Ekeombe keria ekenene getenenerete obokoreri bwechinyomba chiokwegendereria chiokwegendereria chiokwegendereria chiokwegendereria chiokwegendereria chiokwegende |
| Health | Beka ekeombe keria ekiagera na gosaba ng'a ebiro bionsi bionsi bionsi bionsi bionsi bionsi bionsi bionsi bionsi bionsi bionsi bionsi bionsi bionsi bionsi bionsi |
| Agriculture | Ogoikerania obokoreri: Ogoikerania obokoreri bwoboremi goikerania obokoreri obuya, obokoreri obuya, obokoreri obuya na obokoreri obokoreri obokoreri obokoreri o |
| Agriculture | Obotekinorochia bwoboboremi bwegosaba buya ase okorwania oboremi obuya ase ebinyoro bia Afrika ase endagera ye'Sahara. Obotekinorochia bw'obotekinorochia bw'obo |
| Agriculture | Rora ebinto biria biria bikorigereria ebinto biria bikorigereria ebinto biria bikorigereria ebinto biria bikoremereria ebinto biria bikoremereria ebinto biria b |
| Security | Ebinyoro biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria biria b |

## guz-en  (n=138)

### Per-domain scores

| Domain | n | BLEU | chrF | Repetition-flagged |
|---|---:|---:|---:|---:|
| Agriculture | 19 | 2.86 | 18.41 | 0 |
| Education | 38 | 3.01 | 20.08 | 0 |
| Governance | 18 | 4.16 | 20.06 | 0 |
| Health | 25 | 4.88 | 21.91 | 0 |
| Security | 38 | 1.78 | 19.58 | 0 |

### 8 lowest-chrF examples

| Domain | chrF | Source | Reference | Model output |
|---|---:|---|---|---|
| Education | 6.6 | Ogosomia goetera egedichitari bosa ochakirwe. | Free national revision TV channel launches Dec 1 – KBC Channel 23, 24/7! | Ogosomia goetera egedichitari bosa okirwe. |
| Education | 11.1 | Egekwegia buna ense ekorigia, kobanga na gotumia chibesa. | Simulates real-life county fiscal choices | Egekwegia buna ense ekorigia, woga and gotumia chibesa. |
| Education | 11.2 | Toma amang'ana ameng'e goetera esimi arengete obogesaku. | SMS-based quizzes with civic themes | Toma touches the goetera and then the obogesaku. |
| Education | 11.2 | Keania ogotareng'ana. Karwe ebigokonya abana goikera etkinorochia. | Bridge the gap! Provide digital tools for disabled students. | Keania ogotareng'ana. Karwe ebigokonya abana goikera etkinorochia. |
| Health | 11.4 | Abana bakongusa amarioki nkonyora bare amarwaire ya amayaa na okoeyana | Children exposed to smoke are more likely to develop asthma and respiratory infections. | Abana bakongusa amarioki nkonyora bare amarwaire ya amaya na okoeyana |
| Education | 11.9 | Kae chisukuru chibesa! Rora ng'a chibesa chiria eserikari ekorwa ase kera omwana chiakeranire gokonya emeroberio yesukur | Fund schools! Ensure timely capitation for stability. | Kae chisukuru chibesa! Rora ng'a chibesa chiria eserikari ekorwa ase kera omwana chiakeranire gokonya emeroberio yesukur |
| Education | 12.2 | Ebiombe korwa amaorokererio ekero kwegwiki yobogesaku | Clubs to perform for civic week | Ebiombe korwa amaorokererio ekero kwegwiki yobogesaku |
| Education | 12.2 | Ribaga riabeene ase abana baria bare nokogania kwabeene. | Priority for bursary to needy students. | The two of us have a lot of friends and get to know each other. |

### Repetition-loop flagged examples (up to 8)

_None flagged in this direction._
