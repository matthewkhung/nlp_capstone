# source: https://gist.github.com/MrEliptik/b3f16179aa2f530781ef8ca9a16499af
# downloaded 3/5/2023 and modified

import re, string, unicodedata
import nltk
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import pandas as pd
from collections import Counter

import logging


# https://gist.github.com/bgusach/a967e0587d6e01e889fd1d776c5f3729
def multireplace(string: str, replacements: str, ignore_case=False) -> str:
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :param bool ignore_case: whether the match should be case insensitive
    :rtype: str
    """
    if not replacements:
        # Edge case that'd produce a funny regex and cause a KeyError
        return string

    # If case insensitive, we need to normalize the old string so that later a replacement
    # can be found. For instance with {"HEY": "lol"} we should match and find a replacement for "hey",
    # "HEY", "hEy", etc.
    if ignore_case:
        def normalize_old(s):
            return s.lower()

        re_mode = re.IGNORECASE

    else:
        def normalize_old(s):
            return s

        re_mode = 0

    replacements = {normalize_old(key): val for key, val in replacements.items()}

    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    rep_sorted = sorted(replacements, key=len, reverse=True)
    rep_escaped = map(re.escape, rep_sorted)

    # Create a big OR regex that matches any of the substrings to replace
    pattern = re.compile("|".join(rep_escaped), re_mode)

    # For each match, look up the new string in the replacements, being the key the normalized old string
    return pattern.sub(lambda match: replacements[normalize_old(match.group(0))], string)


# cleaning help found here to clean non ascii characters, contractions, abbr
# modified from link below to run faster (~30x improvement)
# https://www.kaggle.com/code/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert#5.-Mislabeled-Samples
def clean(tweet: str) -> str:
    # Special characters
    dict_replace = {
        # special characters
        r"\x89Û_": "", r"\x89ÛÒ": "", r"\x89ÛÓ": "", r"\x89ÛÏWhen": "When", r"\x89ÛÏ": "", r"China\x89Ûªs": "China's",
        r"let\x89Ûªs": "let's", r"\x89Û÷": "", r"\x89Ûª": "", r"\x89Û\x9d": "", r"å_": "", r"\x89Û¢": "",
        r"\x89Û¢åÊ": "", r"fromåÊwounds": "from wounds", r"åÊ": "", r"åÈ": "", r"JapÌ_n": "Japan", r"Ì©": "e",
        r"å¨": "", r"SuruÌ¤": "Suruc", r"åÇ": "", r"å£3million": "3 million", r"åÀ": "",
        # contractions
        r"he's": "he is", r"there's": "there is", r"We're": "We are", r"That's": "That is", r"won't": "will not",
        r"they're": "they are", r"Can't": "Cannot", r"wasn't": "was not", r"don\x89Ûªt": "do not", r"aren't": "are not",
        r"isn't": "is not", r"What's": "What is", r"haven't": "have not", r"hasn't": "has not", r"There's": "There is",
        r"He's": "He is", r"It's": "It is", r"You're": "You are", r"I'M": "I am", r"shouldn't": "should not",
        r"wouldn't": "would not", r"i'm": "I am", r"I\x89Ûªm": "I am", r"I'm": "I am", r"Isn't": "is not",
        r"Here's": "Here is", r"you've": "you have", r"you\x89Ûªve": "you have", r"we're": "we are",
        r"what's": "what is", r"couldn't": "could not", r"we've": "we have", r"it\x89Ûªs": "it is",
        r"doesn\x89Ûªt": "does not", r"It\x89Ûªs": "It is", r"Here\x89Ûªs": "Here is", r"who's": "who is",
        r"I\x89Ûªve": "I have", r"y'all": "you all", r"can\x89Ûªt": "cannot", r"would've": "would have",
        r"it'll": "it will", r"we'll": "we will", r"wouldn\x89Ûªt": "would not", r"We've": "We have",
        r"he'll": "he will", r"Y'all": "You all", r"Weren't": "Were not", r"Didn't": "Did not", r"they'll": "they will",
        r"they'd": "they would", r"DON'T": "DO NOT", r"That\x89Ûªs": "That is", r"they've": "they have",
        r"i'd": "I would", r"should've": "should have", r"You\x89Ûªre": "You are", r"where's": "where is",
        r"Don\x89Ûªt": "Do not", r"we'd": "we would", r"i'll": "I will", r"weren't": "were not", r"They're": "They are",
        r"Can\x89Ûªt": "Cannot", r"you\x89Ûªll": "you will", r"I\x89Ûªd": "I would", r"let's": "let us",
        r"it's": "it is", r"can't": "cannot", r"don't": "do not", r"you're": "you are", r"i've": "I have",
        r"that's": "that is", r"i'll": "I will", r"doesn't": "does not", r"i'd": "I would", r"didn't": "did not",
        r"ain't": "am not", r"you'll": "you will", r"I've": "I have", r"Don't": "do not", r"I'll": "I will",
        r"I'd": "I would", r"Let's": "Let us", r"you'd": "You would", r"It's": "It is", r"Ain't": "am not",
        r"Haven't": "Have not", r"Could've": "Could have", r"youve": "you have", r"donå«t": "do not",
        # Character entity references
        r"&gt;": ">", r"&lt;": "<", r"&amp;": "&",
        # Typos, slang and informal abbreviations
        r"w/e": "whatever", r"w/": "with", r"USAgov": "USA government", r"recentlu": "recently", r"Ph0tos": "Photos",
        r"amirite": "am I right", r"exp0sed": "exposed", r"<3": "love", r"amageddon": "armageddon", r"Trfc": "Traffic",
        r"8/5/2015": "2015-08-05", r"WindStorm": "Wind Storm", r"8/6/2015": "2015-08-06", r"10:38PM": "10:38 PM",
        r"10:30pm": "10:30 PM", r"16yr": "16 year", r"lmao": "laughing my ass off", r"TRAUMATISED": "traumatized",
        # Hashtags and usernames
        r"IranDeal": "Iran Deal", r"ArianaGrande": "Ariana Grande", r"camilacabello97": "camila cabello",
        r"RondaRousey": "Ronda Rousey", r"MTVHottest": "MTV Hottest", r"TrapMusic": "Trap Music",
        r"ProphetMuhammad": "Prophet Muhammad", r"PantherAttack": "Panther Attack",
        r"StrategicPatience": "Strategic Patience", r"socialnews": "social news", r"NASAHurricane": "NASA Hurricane",
        r"onlinecommunities": "online communities", r"humanconsumption": "human consumption",
        r"Typhoon-Devastated": "Typhoon Devastated", r"Meat-Loving": "Meat Loving", r"facialabuse": "facial abuse",
        r"LakeCounty": "Lake County", r"BeingAuthor": "Being Author", r"withheavenly": "with heavenly",
        r"thankU": "thank you", r"iTunesMusic": "iTunes Music", r"OffensiveContent": "Offensive Content",
        r"WorstSummerJob": "Worst Summer Job", r"HarryBeCareful": "Harry Be Careful",
        r"NASASolarSystem": "NASA Solar System", r"animalrescue": "animal rescue", r"KurtSchlichter": "Kurt Schlichter",
        r"aRmageddon": "armageddon", r"Throwingknifes": "Throwing knives", r"GodsLove": "God's Love",
        r"bookboost": "book boost", r"ibooklove": "I book love", r"NestleIndia": "Nestle India",
        r"realDonaldTrump": "Donald Trump", r"DavidVonderhaar": "David Vonderhaar", r"CecilTheLion": "Cecil The Lion",
        r"weathernetwork": "weather network", r"withBioterrorism&use": "with Bioterrorism & use",
        r"Hostage&2": "Hostage & 2", r"GOPDebate": "GOP Debate", r"RickPerry": "Rick Perry", r"frontpage": "front page",
        r"NewsInTweets": "News In Tweets", r"ViralSpell": "Viral Spell", r"til_now": "until now",
        r"volcanoinRussia": "volcano in Russia", r"ZippedNews": "Zipped News", r"MicheleBachman": "Michele Bachman",
        r"53inch": "53 inch", r"KerrickTrial": "Kerrick Trial", r"abstorm": "Alberta Storm", r"Beyhive": "Beyonce hive",
        r"IDFire": "Idaho Fire", r"DETECTADO": "Detected", r"RockyFire": "Rocky Fire", r"Listen/Buy": "Listen / Buy",
        r"NickCannon": "Nick Cannon", r"FaroeIslands": "Faroe Islands", r"yycstorm": "Calgary Storm",
        r"IDPs:": "Internally Displaced People :", r"ArtistsUnited": "Artists United",
        r"ClaytonBryant": "Clayton Bryant", r"jimmyfallon": "jimmy fallon", r"justinbieber": "justin bieber",
        r"UTC2015": "UTC 2015", r"Time2015": "Time 2015", r"djicemoon": "dj icemoon", r"LivingSafely": "Living Safely",
        r"FIFA16": "Fifa 2016", r"thisiswhywecanthavenicethings": "this is why we cannot have nice things",
        r"bbcnews": "bbc news", r"UndergroundRailraod": "Underground Railraod", r"c4news": "c4 news",
        r"OBLITERATION": "obliteration", r"MUDSLIDE": "mudslide", r"NoSurrender": "No Surrender",
        r"NotExplained": "Not Explained", r"greatbritishbakeoff": "great british bake off",
        r"LondonFire": "London Fire", r"KOTAWeather": "KOTA Weather", r"LuchaUnderground": "Lucha Underground",
        r"KOIN6News": "KOIN 6 News", r"LiveOnK2": "Live On K2", r"9NewsGoldCoast": "9 News Gold Coast",
        r"nikeplus": "nike plus", r"david_cameron": "David Cameron", r"peterjukes": "Peter Jukes",
        r"JamesMelville": "James Melville", r"megynkelly": "Megyn Kelly", r"cnewslive": "C News Live",
        r"JamaicaObserver": "Jamaica Observer",
        r"TweetLikeItsSeptember11th2001": "Tweet like it is september 11th 2001", r"cbplawyers": "cbp lawyers",
        r"fewmoretweets": "few more tweets", r"BlackLivesMatter": "Black Lives Matter", r"cjoyner": "Chris Joyner",
        r"ENGvAUS": "England vs Australia", r"ScottWalker": "Scott Walker", r"MikeParrActor": "Michael Parr",
        r"4PlayThursdays": "Foreplay Thursdays", r"TGF2015": "Tontitown Grape Festival", r"realmandyrain": "Mandy Rain",
        r"GraysonDolan": "Grayson Dolan", r"ApolloBrown": "Apollo Brown", r"saddlebrooke": "Saddlebrooke",
        r"TontitownGrape": "Tontitown Grape", r"AbbsWinston": "Abbs Winston", r"ShaunKing": "Shaun King",
        r"MeekMill": "Meek Mill", r"TornadoGiveaway": "Tornado Giveaway", r"GRupdates": "GR updates",
        r"SouthDowns": "South Downs", r"braininjury": "brain injury", r"auspol": "Australian politics",
        r"PlannedParenthood": "Planned Parenthood", r"calgaryweather": "Calgary Weather",
        r"weallheartonedirection": "we all heart one direction", r"edsheeran": "Ed Sheeran",
        r"TrueHeroes": "True Heroes", r"S3XLEAK": "sex leak", r"ComplexMag": "Complex Magazine",
        r"TheAdvocateMag": "The Advocate Magazine", r"CityofCalgary": "City of Calgary",
        r"EbolaOutbreak": "Ebola Outbreak", r"SummerFate": "Summer Fate", r"RAmag": "Royal Academy Magazine",
        r"offers2go": "offers to go", r"foodscare": "food scare",
        r"MNPDNashville": "Metropolitan Nashville Police Department", r"TfLBusAlerts": "TfL Bus Alerts",
        r"GamerGate": "Gamer Gate", r"IHHen": "Humanitarian Relief", r"spinningbot": "spinning bot",
        r"ModiMinistry": "Modi Ministry", r"TAXIWAYS": "taxi ways", r"Calum5SOS": "Calum Hood", r"po_st": "po.st",
        r"scoopit": "scoop.it", r"UltimaLucha": "Ultima Lucha", r"JonathanFerrell": "Jonathan Ferrell",
        r"aria_ahrary": "Aria Ahrary", r"rapidcity": "Rapid City", r"OutBid": "outbid",
        r"lavenderpoetrycafe": "lavender poetry cafe", r"EudryLantiqua": "Eudry Lantiqua", r"15PM": "15 PM",
        r"OriginalFunko": "Funko", r"rightwaystan": "Richard Tan", r"CindyNoonan": "Cindy Noonan",
        r"RT_America": "RT America", r"narendramodi": "Narendra Modi", r"BakeOffFriends": "Bake Off Friends",
        r"TeamHendrick": "Hendrick Motorsports", r"alexbelloli": "Alex Belloli", r"itsjustinstuart": "Justin Stuart",
        r"gunsense": "gun sense", r"DebateQuestionsWeWantToHear": "debate questions we want to hear",
        r"RoyalCarribean": "Royal Carribean", r"samanthaturne19": "Samantha Turner", r"JonVoyage": "Jon Stewart",
        r"renew911health": "renew 911 health", r"SuryaRay": "Surya Ray", r"pattonoswalt": "Patton Oswalt",
        r"minhazmerchant": "Minhaz Merchant", r"TLVFaces": "Israel Diaspora Coalition", r"pmarca": "Marc Andreessen",
        r"pdx911": "Portland Police", r"jamaicaplain": "Jamaica Plain", r"Japton": "Arkansas",
        r"RouteComplex": "Route Complex", r"INSubcontinent": "Indian Subcontinent",
        r"NJTurnpike": "New Jersey Turnpike", r"Politifiact": "PolitiFact", r"Hiroshima70": "Hiroshima",
        r"GMMBC": "Greater Mt Moriah Baptist Church", r"versethe": "verse the", r"TubeStrike": "Tube Strike",
        r"MissionHills": "Mission Hills", r"ProtectDenaliWolves": "Protect Denali Wolves", r"NANKANA": "Nankana",
        r"SAHIB": "Sahib", r"PAKPATTAN": "Pakpattan", r"Newz_Sacramento": "News Sacramento", r"gofundme": "go fund me",
        r"pmharper": "Stephen Harper", r"IvanBerroa": "Ivan Berroa", r"LosDelSonido": "Los Del Sonido",
        r"bancodeseries": "banco de series", r"timkaine": "Tim Kaine", r"IdentityTheft": "Identity Theft",
        r"AllLivesMatter": "All Lives Matter", r"mishacollins": "Misha Collins", r"BillNeelyNBC": "Bill Neely",
        r"BeClearOnCancer": "be clear on cancer", r"Kowing": "Knowing", r"ScreamQueens": "Scream Queens",
        r"AskCharley": "Ask Charley", r"BlizzHeroes": "Heroes of the Storm", r"BradleyBrad47": "Bradley Brad",
        r"HannaPH": "Typhoon Hanna", r"meinlcymbals": "MEINL Cymbals", r"Ptbo": "Peterborough",
        r"cnnbrk": "CNN Breaking News", r"IndianNews": "Indian News", r"savebees": "save bees",
        r"GreenHarvard": "Green Harvard", r"StandwithPP": "Stand with planned parenthood",
        r"hermancranston": "Herman Cranston", r"WMUR9": "WMUR-TV", r"RockBottomRadFM": "Rock Bottom Radio",
        r"ameenshaikh3": "Ameen Shaikh", r"ProSyn": "Project Syndicate", r"Daesh": "ISIS", r"s2g": "swear to god",
        r"listenlive": "listen live", r"CDCgov": "Centers for Disease Control and Prevention", r"FoxNew": "Fox News",
        r"CBSBigBrother": "Big Brother", r"JulieDiCaro": "Julie DiCaro", r"theadvocatemag": "The Advocate Magazine",
        r"RohnertParkDPS": "Rohnert Park Police Department", r"THISIZBWRIGHT": "Bonnie Wright",
        r"Popularmmos": "Popular MMOs", r"WildHorses": "Wild Horses", r"FantasticFour": "Fantastic Four",
        r"HORNDALE": "Horndale", r"PINER": "Piner", r"BathAndNorthEastSomerset": "Bath and North East Somerset",
        r"thatswhatfriendsarefor": "that is what friends are for", r"residualincome": "residual income",
        r"YahooNewsDigest": "Yahoo News Digest", r"MalaysiaAirlines": "Malaysia Airlines",
        r"AmazonDeals": "Amazon Deals", r"MissCharleyWebb": "Charley Webb", r"shoalstraffic": "shoals traffic",
        r"GeorgeFoster72": "George Foster", r"pop2015": "pop 2015", r"_PokemonCards_": "Pokemon Cards",
        r"DianneG": "Dianne Gallagher", r"KashmirConflict": "Kashmir Conflict", r"BritishBakeOff": "British Bake Off",
        r"FreeKashmir": "Free Kashmir", r"mattmosley": "Matt Mosley", r"BishopFred": "Bishop Fred",
        r"EndConflict": "End Conflict", r"EndOccupation": "End Occupation", r"UNHEALED": "unhealed",
        r"CharlesDagnall": "Charles Dagnall", r"Latestnews": "Latest news", r"KindleCountdown": "Kindle Countdown",
        r"NoMoreHandouts": "No More Handouts", r"datingtips": "dating tips", r"charlesadler": "Charles Adler",
        r"twia": "Texas Windstorm Insurance Association", r"txlege": "Texas Legislature",
        r"WindstormInsurer": "Windstorm Insurer", r"Newss": "News", r"hempoil": "hemp oil",
        r"CommoditiesAre": "Commodities are", r"tubestrike": "tube strike", r"JoeNBC": "Joe Scarborough",
        r"LiteraryCakes": "Literary Cakes", r"TI5": "The International 5", r"thehill": "the hill",
        r"3others": "3 others", r"stighefootball": "Sam Tighe",
        r"whatstheimportantvideo": "what is the important video", r"ClaudioMeloni": "Claudio Meloni",
        r"DukeSkywalker": "Duke Skywalker", r"carsonmwr": "Fort Carson", r"offdishduty": "off dish duty",
        r"andword": "and word", r"rhodeisland": "Rhode Island", r"easternoregon": "Eastern Oregon",
        r"WAwildfire": "Washington Wildfire", r"fingerrockfire": "Finger Rock Fire", r"57am": "57 am",
        r"fingerrockfire": "Finger Rock Fire", r"JacobHoggard": "Jacob Hoggard", r"newnewnew": "new new new",
        r"under50": "under 50", r"getitbeforeitsgone": "get it before it is gone",
        r"freshoutofthebox": "fresh out of the box", r"amwriting": "am writing", r"Bokoharm": "Boko Haram",
        r"Nowlike": "Now like", r"seasonfrom": "season from", r"epicente": "epicenter", r"epicenterr": "epicenter",
        r"sicklife": "sick life", r"yycweather": "Calgary Weather", r"calgarysun": "Calgary Sun",
        r"approachng": "approaching", r"evng": "evening", r"Sumthng": "something", r"EllenPompeo": "Ellen Pompeo",
        r"shondarhimes": "Shonda Rhimes", r"ABCNetwork": "ABC Network", r"SushmaSwaraj": "Sushma Swaraj",
        r"pray4japan": "Pray for Japan", r"hope4japan": "Hope for Japan", r"Illusionimagess": "Illusion images",
        r"SummerUnderTheStars": "Summer Under The Stars", r"ShallWeDance": "Shall We Dance", r"TCMParty": "TCM Party",
        r"marijuananews": "marijuana news", r"onbeingwithKristaTippett": "on being with Krista Tippett",
        r"Beingtweets": "Being tweets", r"newauthors": "new authors", r"remedyyyy": "remedy", r"44PM": "44 PM",
        r"HeadlinesApp": "Headlines App", r"40PM": "40 PM", r"myswc": "Severe Weather Center", r"ithats": "that is",
        r"icouldsitinthismomentforever": "I could sit in this moment forever", r"FatLoss": "Fat Loss", r"02PM": "02 PM",
        r"MetroFmTalk": "Metro Fm Talk", r"Bstrd": "bastard", r"bldy": "bloody", r"MetrofmTalk": "Metro Fm Talk",
        r"terrorismturn": "terrorism turn", r"BBCNewsAsia": "BBC News Asia", r"BehindTheScenes": "Behind The Scenes",
        r"GeorgeTakei": "George Takei", r"WomensWeeklyMag": "Womens Weekly Magazine",
        r"SurvivorsGuidetoEarth": "Survivors Guide to Earth", r"incubusband": "incubus band",
        r"Babypicturethis": "Baby picture this", r"BombEffects": "Bomb Effects", r"win10": "Windows 10",
        r"idkidk": "I do not know I do not know", r"TheWalkingDead": "The Walking Dead", r"amyschumer": "Amy Schumer",
        r"crewlist": "crew list", r"Erdogans": "Erdogan", r"BBCLive": "BBC Live", r"TonyAbbottMHR": "Tony Abbott",
        r"paulmyerscough": "Paul Myerscough", r"georgegallagher": "George Gallagher",
        r"JimmieJohnson": "Jimmie Johnson", r"pctool": "pc tool", r"DoingHashtagsRight": "Doing Hashtags Right",
        r"ThrowbackThursday": "Throwback Thursday", r"SnowBackSunday": "Snowback Sunday", r"LakeEffect": "Lake Effect",
        r"RTphotographyUK": "Richard Thomas Photography UK", r"BigBang_CBS": "Big Bang CBS",
        r"writerslife": "writers life", r"NaturalBirth": "Natural Birth", r"UnusualWords": "Unusual Words",
        r"wizkhalifa": "Wiz Khalifa", r"acreativedc": "a creative DC", r"vscodc": "vsco DC", r"VSCOcam": "vsco camera",
        r"TheBEACHDC": "The beach DC", r"buildingmuseum": "building museum", r"WorldOil": "World Oil",
        r"redwedding": "red wedding", r"AmazingRaceCanada": "Amazing Race Canada", r"WakeUpAmerica": "Wake Up America",
        r"\\Allahuakbar\\": "Allahu Akbar", r"bleased": "blessed", r"nigeriantribune": "Nigerian Tribune",
        r"HIDEO_KOJIMA_EN": "Hideo Kojima", r"FusionFestival": "Fusion Festival", r"50Mixed": "50 Mixed",
        r"NoAgenda": "No Agenda", r"WhiteGenocide": "White Genocide", r"dirtylying": "dirty lying",
        r"SyrianRefugees": "Syrian Refugees", r"changetheworld": "change the world", r"Ebolacase": "Ebola case",
        r"mcgtech": "mcg technologies", r"withweapons": "with weapons", r"advancedwarfare": "advanced warfare",
        r"letsFootball": "let us Football", r"LateNiteMix": "late night mix", r"PhilCollinsFeed": "Phil Collins",
        r"RudyHavenstein": "Rudy Havenstein", r"22PM": "22 PM", r"54am": "54 AM", r"38am": "38 AM",
        r"OldFolkExplainStuff": "Old Folk Explain Stuff", r"BlacklivesMatter": "Black Lives Matter",
        r"InsaneLimits": "Insane Limits", r"youcantsitwithus": "you cannot sit with us", r"2k15": "2015",
        r"TheIran": "Iran", r"JimmyFallon": "Jimmy Fallon", r"AlbertBrooks": "Albert Brooks",
        r"defense_news": "defense news", r"nuclearrcSA": "Nuclear Risk Control Self Assessment",
        r"Auspol": "Australia Politics", r"NuclearPower": "Nuclear Power", r"WhiteTerrorism": "White Terrorism",
        r"truthfrequencyradio": "Truth Frequency Radio", r"ErasureIsNotEquality": "Erasure is not equality",
        r"ProBonoNews": "Pro Bono News", r"JakartaPost": "Jakarta Post", r"toopainful": "too painful",
        r"melindahaunton": "Melinda Haunton", r"NoNukes": "No Nukes", r"curryspcworld": "Currys PC World",
        r"ineedcake": "I need cake", r"blackforestgateau": "black forest gateau", r"BBCOne": "BBC One",
        r"AlexxPage": "Alex Page", r"jonathanserrie": "Jonathan Serrie", r"SocialJerkBlog": "Social Jerk Blog",
        r"ChelseaVPeretti": "Chelsea Peretti", r"irongiant": "iron giant", r"RonFunches": "Ron Funches",
        r"TimCook": "Tim Cook", r"sebastianstanisaliveandwell": "Sebastian Stan is alive and well",
        r"Madsummer": "Mad summer", r"NowYouKnow": "Now you know", r"concertphotography": "concert photography",
        r"TomLandry": "Tom Landry", r"showgirldayoff": "show girl day off", r"Yougslavia": "Yugoslavia",
        r"QuantumDataInformatics": "Quantum Data Informatics", r"FromTheDesk": "From The Desk",
        r"TheaterTrial": "Theater Trial", r"CatoInstitute": "Cato Institute", r"EmekaGift": "Emeka Gift",
        r"LetsBe_Rational": "Let us be rational", r"Cynicalreality": "Cynical reality",
        r"FredOlsenCruise": "Fred Olsen Cruise", r"NotSorry": "not sorry", r"UseYourWords": "use your words",
        r"WordoftheDay": "word of the day", r"Dictionarycom": "Dictionary.com", r"TheBrooklynLife": "The Brooklyn Life",
        r"jokethey": "joke they", r"nflweek1picks": "NFL week 1 picks", r"uiseful": "useful",
        r"JusticeDotOrg": "The American Association for Justice", r"autoaccidents": "auto accidents",
        r"SteveGursten": "Steve Gursten", r"MichiganAutoLaw": "Michigan Auto Law", r"birdgang": "bird gang",
        r"nflnetwork": "NFL Network", r"NYDNSports": "NY Daily News Sports",
        r"RVacchianoNYDN": "Ralph Vacchiano NY Daily News", r"EdmontonEsks": "Edmonton Eskimos",
        r"david_brelsford": "David Brelsford", r"TOI_India": "The Times of India", r"hegot": "he got",
        r"SkinsOn9": "Skins on 9", r"sothathappened": "so that happened", r"LCOutOfDoors": "LC Out Of Doors",
        r"NationFirst": "Nation First", r"IndiaToday": "India Today", r"HLPS": "helps",
        r"HOSTAGESTHROSW": "hostages throw", r"SNCTIONS": "sanctions", r"BidTime": "Bid Time",
        r"crunchysensible": "crunchy sensible", r"RandomActsOfRomance": "Random acts of romance",
        r"MomentsAtHill": "Moments at hill", r"eatshit": "eat shit", r"liveleakfun": "live leak fun",
        r"SahelNews": "Sahel News", r"abc7newsbayarea": "ABC 7 News Bay Area",
        r"facilitiesmanagement": "facilities management", r"facilitydude": "facility dude",
        r"CampLogistics": "Camp logistics", r"alaskapublic": "Alaska public", r"MarketResearch": "Market Research",
        r"AccuracyEsports": "Accuracy Esports", r"TheBodyShopAust": "The Body Shop Australia",
        r"yychail": "Calgary hail", r"yyctraffic": "Calgary traffic", r"eliotschool": "eliot school",
        r"TheBrokenCity": "The Broken City", r"OldsFireDept": "Olds Fire Department", r"RiverComplex": "River Complex",
        r"fieldworksmells": "field work smells", r"IranElection": "Iran Election", r"glowng": "glowing",
        r"kindlng": "kindling", r"riggd": "rigged", r"slownewsday": "slow news day", r"MyanmarFlood": "Myanmar Flood",
        r"abc7chicago": "ABC 7 Chicago", r"copolitics": "Colorado Politics", r"AdilGhumro": "Adil Ghumro",
        r"netbots": "net bots", r"byebyeroad": "bye bye road", r"massiveflooding": "massive flooding",
        r"EndofUS": "End of United States", r"35PM": "35 PM", r"greektheatrela": "Greek Theatre Los Angeles",
        r"76mins": "76 minutes", r"publicsafetyfirst": "public safety first", r"livesmatter": "lives matter",
        r"myhometown": "my hometown", r"tankerfire": "tanker fire", r"MEMORIALDAY": "memorial day",
        r"MEMORIAL_DAY": "memorial day", r"instaxbooty": "instagram booty", r"Jerusalem_Post": "Jerusalem Post",
        r"WayneRooney_INA": "Wayne Rooney", r"VirtualReality": "Virtual Reality", r"OculusRift": "Oculus Rift",
        r"OwenJones84": "Owen Jones", r"jeremycorbyn": "Jeremy Corbyn", r"paulrogers002": "Paul Rogers",
        r"mortalkombatx": "Mortal Kombat X", r"mortalkombat": "Mortal Kombat", r"FilipeCoelho92": "Filipe Coelho",
        r"OnlyQuakeNews": "Only Quake News", r"kostumes": "costumes", r"YEEESSSS": "yes",
        r"ToshikazuKatayama": "Toshikazu Katayama", r"IntlDevelopment": "Intl Development",
        r"ExtremeWeather": "Extreme Weather", r"WereNotGruberVoters": "We are not gruber voters",
        r"NewsThousands": "News Thousands", r"EdmundAdamus": "Edmund Adamus", r"EyewitnessWV": "Eye witness WV",
        r"PhiladelphiaMuseu": "Philadelphia Museum", r"DublinComicCon": "Dublin Comic Con",
        r"NicholasBrendon": "Nicholas Brendon", r"Alltheway80s": "All the way 80s", r"FromTheField": "From the field",
        r"NorthIowa": "North Iowa", r"WillowFire": "Willow Fire", r"MadRiverComplex": "Mad River Complex",
        r"feelingmanly": "feeling manly", r"stillnotoverit": "still not over it",
        r"FortitudeValley": "Fortitude Valley", r"CoastpowerlineTramTr": "Coast powerline",
        r"ServicesGold": "Services Gold", r"NewsbrokenEmergency": "News broken emergency", r"Evaucation": "evacuation",
        r"leaveevacuateexitbe": "leave evacuate exit be", r"P_EOPLE": "PEOPLE", r"Tubestrike": "tube strike",
        r"CLASS_SICK": "CLASS SICK", r"localplumber": "local plumber", r"awesomejobsiri": "awesome job siri",
        r"PayForItHow": "Pay for it how", r"ThisIsAfrica": "This is Africa", r"crimeairnetwork": "crime air network",
        r"KimAcheson": "Kim Acheson", r"cityofcalgary": "City of Calgary", r"prosyndicate": "pro syndicate",
        r"660NEWS": "660 NEWS", r"BusInsMagazine": "Business Insurance Magazine", r"wfocus": "focus",
        r"ShastaDam": "Shasta Dam", r"go2MarkFranco": "Mark Franco", r"StephGHinojosa": "Steph Hinojosa",
        r"Nashgrier": "Nash Grier", r"NashNewVideo": "Nash new video",
        r"IWouldntGetElectedBecause": "I would not get elected because", r"SHGames": "Sledgehammer Games",
        r"bedhair": "bed hair", r"JoelHeyman": "Joel Heyman", r"viaYouTube": "via YouTube",
        # Acronyms
        r"MH370": "Malaysia Airlines Flight 370", r"mÌ¼sica": "music", r"okwx": "Oklahoma City Weather",
        r"arwx": "Arkansas Weather", r"gawx": "Georgia Weather", r"scwx": "South Carolina Weather",
        r"cawx": "California Weather", r"tnwx": "Tennessee Weather", r"azwx": "Arizona Weather",
        r"alwx": "Alabama Weather", r"wordpressdotcom": "wordpress",
        r"usNWSgov": "United States National Weather Service", r"Suruc": "Sanliurfa",
        # Grouping same words without embeddings
        r"Bestnaijamade": "bestnaijamade", r"SOUDELOR": "Soudelor",
    }

    tweet = multireplace(tweet, dict_replace)

    # Urls
    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)

    # Words with punctuations and special characters
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        tweet = tweet.replace(p, f' {p} ')

    # ... and ..
    tweet = tweet.replace('...', ' ... ')
    if '...' not in tweet:
        tweet = tweet.replace('..', ' ... ')

    return tweet


def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words


# def preprocess(sample):
#     sample = remove_URL(sample)
#     sample = replace_contractions(sample)
#     # Tokenize
#     words = nltk.word_tokenize(sample)
#
#     # Normalize
#     return normalize(words)


# build custom preprocessor and custom tokenizer functions
def preprocessor(text):
    text = clean(text)
#    text = remove_URL(text)
    return text


def tokenizer(text):
    tokenizer = TweetTokenizer()
    words = tokenizer.tokenize(text)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words


def count_words_from_series(ds_text: pd.Series) -> pd.DataFrame:
    # join all texts into one string
    text = ' '.join(i for i in ds_text)

    # preprocess and tokenize text
    text = preprocessor(text)
    words = tokenizer(text)

    # build frequency map
    df_words = pd.DataFrame.from_dict(Counter(words), orient='index').rename(columns={0: 'count'})
    df_words.sort_values(by='count', ascending=False, inplace=True)
    logging.info(f'Unique words: {len(df_words)}')
    return df_words


if __name__ == "__main__":
    sample = "Blood test for Down's syndrome hailed http://bbc.in/1BO3eWQ"

    sample = remove_URL(sample)
    sample = replace_contractions(sample)

    # Tokenize
    words = nltk.word_tokenize(sample)
    print(words)

    # Normalize
    words = normalize(words)
    print(words)
