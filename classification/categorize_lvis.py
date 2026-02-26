import json

# LVIS Taxonomy 정의 (다른 모듈에서 참조 가능하도록 외부로 추출)
LVIS_TAXONOMY_DICT = {
    "Animals & Pets": [
        "animal", "dog", "cat", "bird", "fish", "horse", "bear", "lion", "tiger", "elephant", "insect", "monkey", "reptile", "snake", "mouse", "rat", "pet", "alligator", "baboon", "calf", "camel", "cow", "crab", "dalmatian", "deer", "dolphin", "duck", "eagle", "elk", "falcon", "ferret", "flamingo", "frog", "gazelle", "giant_panda", "giraffe", "goat", "goldfish", "goose", "gorilla", "grizzly", "hamster", "heron", "hippopotamus", "hog", "hummingbird", "kitten", "koala", "lamb", "lizard", "monkey", "owl", "parakeet", "parrot", "pelican", "penguin", "pigeon", "polar_bear", "pony", "rabbit", "rhinoceros", "seabird", "seahorse", "shark", "sheep", "shepherd_dog", "snake", "spider", "squirrel", "starfish", "turtle", "walrus", "wolf", "zebra", "bat", "cub", "puppy", "butterfly", "cockroach", "dragonfly", "ladybug", "alligator", "beetle", "bull", "cock", "duckling", "eagle", "falcon", "ferret", "goose", "gull", "heron", "hornet", "hummingbird", "lizard", "mammoth", "manatee", "mouse", "octopus", "ostrich", "owl", "panda", "parakeet", "parrot", "penguin", "pigeon", "rabbit", "rat", "rhinoceros", "seahorse", "shark", "sheep", "snake", "spider", "squirrel", "starfish", "tiger", "turtle", "vulture", "walrus", "wolf", "zebra", "bee", "bug", "worm", "bait", "domestic_ass", "foal", "mallard", "puffer", "puffin", "pug", "slug", "snail", "steer", "stork", "swan", "tuna", "wasp", "whale", "yak", "alligator", "baboon", "badger", "basenji", "beagle", "beaver", "bison", "boar", "boxer", "canary", "chameleon", "cheetah", "chimpanzee", "chinchilla", "chipmunk", "cobra", "collie", "coyote", "crow", "dachshund", "dalmatian", "dingo", "donkey", "dormouse", "dove", "eel", "elk", "emu", "ewe", "falcon", "ferret", "finch", "flamingo", "fox", "frog", "gazelle", "gecko", "gerbil", "gibbon", "giraffe", "goat", "goldfish", "goose", "gorilla", "greyhound", "guinea_pig", "gull", "hamster", "hare", "hawk", "hedgehog", "heron", "hippopotamus", "hog", "horse", "hound", "hummingbird", "hyena", "ibex", "iguana", "impala", "jackal", "jaguar", "kangaroo", "kitten", "koala", "lamb", "lemur", "leopard", "lion", "lizard", "llama", "lynx", "macaque", "magpie", "mallard", "mamoth", "manatee", "mandrill", "marmoset", "marten", "mink", "mole", "mongoose", "monkey", "moose", "mouse", "mule", "newt", "nightingale", "ocelot", "octopus", "okapi", "opossum", "orangutan", "orca", "ostrich", "otter", "owl", "ox", "oyster", "panda", "panther", "parakeet", "parrot", "peacock", "pelican", "penguin", "pig", "pigeon", "platypus", "polar_bear", "pony", "poodle", "porcupine", "porpoise", "puma", "python", "quail", "rabbit", "raccoon", "ram", "rat", "raven", "reindeer", "rhinoceros", "robin", "rooster", "salamander", "salmon", "seal", "sea_lion", "shark", "sheep", "shrew", "shrimp", "skunk", "sloth", "snail", "snake", "sparrow", "spider", "squid", "squirrel", "stallion", "starfish", "stork", "swan", "tapir", "tarantula", "termite", "tiger", "toad", "toucan", "trout", "turkey", "turtle", "viper", "vulture", "walrus", "wasp", "weasel", "whale", "wolf", "wolverine", "wombat", "woodpecker", "worm", "yak", "zebra"
    ],
    "Architecture": [
        "house", "building", "tower", "bridge", "arch", "column", "door", "window", "fence", "gate", "roof", "stairs", "wall", "lighthouse", "clock_tower", "barn", "cabin", "gazebo", "pavilion", "pyramid", "skyscraper", "steeple", "windmill", "cornice", "manhole", "fireplug", "awning", "billboard", "brick", "chimney", "cornet", "fountain", "garage", "mailbox", "monument", "pillar", "postbox", "railing", "skylight", "statue", "vent", "awning", "cornice", "doorknob", "latch", "hinge", "barricade", "cabana", "cistern", "dome", "fire_alarm", "fireplace", "mast", "parapet", "shutter", "silo", "tavern", "tile", "arch", "balcony", "beam", "cabin", "chapel", "column", "cottage", "cupola", "dwelling", "facade", "fire_escape", "foundation", "gutter", "hearth", "loft", "mansion", "niche", "orphanage", "palace", "patio", "porch", "shrine", "spire", "stable", "structure", "temple", "vault", "villa"
    ],
    "Art & Abstract": [
        "sculpture", "statue", "painting", "illustration", "artifact", "totem", "figurine", "mask", "gargoyle", "ornament", "award", "brass_plaque", "ceremonial", "totem_pole", "easel", "palette", "sculpt_tool", "canvas", "frame", "mosaic", "bead", "mosaic", "origami", "pottery", "tapestry", "statuette", "blueprint", "scroll", "artifact", "carving", "ceramics", "design", "emblem", "engraving", "fresco", "icon", "medallion", "mural", "relief", "sketch", "stencil", "tracing"
    ],
    "Cars & Vehicles": [
        "car", "truck", "van", "bus", "train", "airplane", "bicycle", "motorcycle", "boat", "ship", "scooter", "aircraft", "ambulance", "barge", "cab", "camper", "cargo_ship", "convertible", "dirt_bike", "ferry", "fire_engine", "forklift", "freight_car", "garbage_truck", "helicopter", "jeep", "kayak", "limousine", "minivan", "motor_scooter", "motor_vehicle", "pickup_truck", "race_car", "raft", "railcar", "river_boat", "school_bus", "snowmobile", "space_shuttle", "stagecoach", "tractor", "trailer_truck", "unicycle", "yacht", "balloon", "blimp", "canoe", "dinghy", "gondola", "houseboat", "parasail", "raft", "unicycle", "barge", "buoy", "cart", "chariot", "ferry", "jet", "plane", "propeller", "scooter", "submarine", "taxi", "wagon", "wheel", "axel", "bulldozer", "chariot", "golfcart", "oar", "paddle", "seaplane", "tow_truck", "trailer", "tricycle", "unicycle", "vessel"
    ],
    "Characters & Creatures": [
        "character", "creature", "monster", "dragon", "robot", "puppet", "doll", "action_figure", "dummy", "mascot", "mannequin", "teddy_bear", "rag_doll", "dollhouse", "hero", "villain", "alien", "gnome", "troll", "fairy", "avatar", "beast", "cyborg", "mutant", "ogre", "phantom", "spirit", "wraith"
    ],
    "Cultural Heritage & History": [
        "ancient", "medieval", "vintage", "antique", "relic", "historical", "armor", "shield", "sword", "crown", "chalice", "scroll", "tapestry", "gravestone", "urn", "milestone", "parchment", "crucifix", "tiara", "artifact", "totem", "heirloom", "sarcophagus", "amulet", "scroll", "quill", "ancient_monument", "hieroglyphics", "manuscript", "obelisk", "ruins", "totem", "artifact", "treasure"
    ],
    "Electronics & Gadgets": [
        "computer", "phone", "television", "radio", "camera", "tablet", "monitor", "keyboard", "mouse", "laptop", "printer", "scanner", "router", "modem", "remote_control", "ipod", "beeper", "camcorder", "calculator", "gameboard", "joystick", "stereo", "subwoofer", "webcam", "antenna", "battery", "cable", "cassette", "circuit", "control", "earphone", "flashlight", "headset", "microphone", "projector", "speaker", "stylus", "videotape", "cd_player", "record_player", "air_conditioner", "blender", "mixer", "food_processor", "toaster", "vacuum", "igniter", "radar", "thermostat", "videotape", "beeper", "buzzer", "cpu", "display", "hard_drive", "modem", "motherboard", "processor", "sensor", "terminal", "transistor"
    ],
    "Fashion & Style": [
        "dress", "shirt", "pants", "hat", "shoes", "jacket", "coat", "suit", "fashion", "apparel", "uniform", "skirt", "apron", "belt", "blazer", "blouse", "boot", "cap", "cardigan", "costume", "glove", "jean", "jersey", "kimono", "necktie", "pajamas", "parka", "robe", "scarf", "sock", "sweater", "sweatshirt", "swimsuit", "trousers", "vest", "jewelry", "bracelet", "earring", "necklace", "ring", "watch", "handbag", "wallet", "sunglasses", "hair_dryer", "shaver", "perfume", "lipstick", "backpack", "briefcase", "satchel", "tote_bag", "clutch", "purse", "pouch", "bandanna", "beanie", "beret", "bonnet", "bow_tie", "brooch", "buckle", "button", "diaper", "eyepatch", "hairbrush", "handkerchief", "helmet", "mask", "mitten", "necktie", "overalls", "poncho", "slipper", "tights", "turban", "underwear", "veil", "wig", "wristband", "anklet", "armband", "beret", "brassiere", "cincture", "clasp", "cloak", "clutch", "corset", "cufflink", "fleece", "flip_flop", "flipper", "hairnet", "hand_glass", "pantyhose", "shawl", "tiara", "tux", "visor", "wristlet", "brocade", "choker", "clogs", "ensemble", "garment", "headband", "pendant", "raiment", "sneaker", "wardrobe"
    ],
    "Food & Drink": [
        "food", "drink", "meat", "fruit", "vegetable", "bread", "cake", "cookie", "candy", "beverage", "alcohol", "wine", "beer", "juice", "milk", "soda", "apple", "banana", "berry", "orange", "grape", "lemon", "peach", "pear", "potato", "tomato", "carrot", "onion", "garlic", "pizza", "burger", "sandwich", "sushi", "pasta", "sausage", "ham", "steak", "egg", "butter", "honey", "nut", "chocolate", "dessert", "snack", "pretzel", "popcorn", "taco", "burrito", "waffle", "pancake", "muffin", "doughnut", "tea", "coffee", "cider", "liquor", "tequila", "vodka", "yogurt", "soup", "stew", "salad", "salsa", "almond", "apricot", "artichoke", "asparagus", "avocado", "bagel", "baguette", "blackberry", "blueberry", "broccoli", "brownie", "bun", "cabbage", "cantaloupe", "cappuccino", "caramel", "cauliflower", "celery", "cheese", "cherry", "chili", "cocoa", "coconut", "corn", "cracker", "cucumber", "date", "eggplant", "fig", "ginger", "grapefruit", "ham", "hamburger", "ice_cream", "jam", "lasagna", "lemonade", "lettuce", "lime", "lollipop", "marshmallow", "melon", "mushroom", "noodle", "olive", "omelet", "onion", "papaya", "pastry", "pea", "pepper", "pickle", "pie", "pineapple", "pistachio", "plum", "pomegranate", "pork", "pudding", "pumpkin", "radish", "raspberry", "rice", "salami", "salt", "sauce", "shrimp", "spinach", "squid", "strawberry", "sugar", "sweet_potato", "tart", "toast", "tofu", "tomato", "truffle", "turkey", "turnip", "vanilla", "vinegar", "watermelon", "whiskey", "zucchini", "baguet", "beef", "bubble_gum", "clementine", "crouton", "eclair", "edible_corn", "gelatin", "green_bean", "grits", "honey", "icecream", "legume", "meatball", "milkshake", "omelet", "prawn", "prune", "quesadilla", "quiche", "salmon", "sherbert", "smoothie", "sugar", "taco", "tortilla", "truffle", "whipped_cream", "beverage", "breadstick", "broth", "chowder", "condiment", "cuisine", "delicacy", "dish", "entree", "feast", "grocery", "ingredient", "main_course", "morsel", "nosh", "refreshment", "viand"
    ],
    "Furniture & Home": [
        "furniture", "chair", "table", "bed", "sofa", "desk", "closet", "cabinet", "shelf", "lamp", "mirror", "sink", "toilet", "bathtub", "shower", "appliance", "refrigerator", "oven", "microwave", "dishwasher", "washer", "dryer", "drawer", "dresser", "couch", "ottoman", "armchair", "bookcase", "nightstand", "wardrobe", "curtain", "rug", "towel", "bedspread", "pillow", "blanket", "cushion", "clipping", "bucket", "canister", "basket", "crate", "box", "chest", "clock", "fan", "heater", "ironing_board", "sewing_machine", "vacuum_cleaner", "waste_container", "ashtray", "basin", "bowl", "broom", "candle", "carpet", "chandelier", "coaster", "container", "cup", "cupboard", "dish", "dispenser", "doormat", "dustpan", "flask", "fork", "glass", "jar", "kettle", "knife", "ladle", "mop", "mug", "napkin", "pan", "pitcher", "plate", "pot", "saucer", "scraper", "shaker", "spoon", "stool", "strainer", "tray", "vase", "broom", "cloth", "sponge", "soap", "toothpaste", "toothbrush", "tissue", "razor", "aerosol_can", "armoire", "barrel", "bath_mat", "beanbag", "bedpan", "bench", "bottle", "can", "canteen", "carton", "cistern", "coffeepot", "cork", "corkscrew", "crib", "cylinder", "detergent", "dishrag", "dishtowel", "dispenser", "eggbeater", "fireplace", "flowerpot", "funnel", "hammock", "hamper", "keg", "lantern", "loveseat", "mattress", "mop", "platter", "potholder", "recliner", "rolling_pin", "saucepan", "soap", "spatula", "stove", "tank", "teakettle", "teapot", "thermos", "thimble", "tissue", "tongs", "tray", "washbasin", "water_cooler", "wine_bucket", "wok", "wreath", "artifact", "ashtray", "basin", "bowl", "bucket", "casket", "chamber", "container", "cup", "fixture", "furnishing", "hook", "kettle", "knob", "lid", "ornament", "pan", "pitcher", "platter", "pot", "receptacle", "shelf", "table", "urn", "vessel", "kitchenware", "household", "tableware"
    ],
    "Music": [
        "music", "instrument", "guitar", "piano", "drum", "violin", "flute", "trumpet", "saxophone", "accordion", "banjo", "bass", "cello", "clarinet", "harp", "organ", "tambourine", "ukulele", "harmonium", "metronome", "music_stool", "gramophone", "record_player", "speaker", "amplifier", "bell", "chime", "cymbal", "gong", "horn", "key", "lyre", "maraca", "oboe", "whistle", "triange", "cornet", "drumstick", "composition", "melody", "notation", "orchestra", "rhythm", "serenade", "symphony"
    ],
    "Nature & Plants": [
        "nature", "plant", "tree", "flower", "leaf", "bush", "grass", "mountain", "rock", "stone", "river", "beach", "sand", "soil", "log", "stump", "twig", "branch", "cone", "cactus", "fern", "moss", "bamboo", "sunflower", "rose", "tulip", "daisy", "pinecone", "seashell", "moss", "mushroom", "ocean", "reef", "shell", "vine", "wood", "forest", "jungle", "swamp", "aquarium", "birdbath", "birdcage", "birdfeeder", "birdhouse", "flowerpot", "gourd", "log", "nest", "shell", "flora", "foliage", "grove", "orchard", "seedling", "shrub", "thicket", "vegetation"
    ],
    "News & Politics": [
        "news", "politics", "newspaper", "magazine", "billboard", "poster", "advertisement", "flag", "banner", "sign", "ballot", "badge", "diploma", "manifesto", "record", "ledger", "bulletin", "ticket", "stamp", "receipt", "envelope", "identity_card", "passport", "board", "book", "booklet", "bookmark", "card", "calendar", "checkbook", "diary", "dollar", "inkpad", "ledger", "map", "marker", "money", "notebook", "notepad", "pennant", "postcard", "tag", "affidavit", "brochure", "charter", "document", "edict", "gazette", "journal", "pamplet", "treaty"
    ],
    "People": [
        "person", "human", "man", "woman", "child", "baby", "athlete", "doctor", "firefighter", "police", "soldier", "teacher", "worker", "crowd", "identity_card", "passport", "stroller", "pacifier", "highchair", "diaper", "statue", "suit", "baby_buggy", "crutch", "stretcher", "anatomy", "citizen", "individual", "mortal"
    ],
    "Places & Locations": [
        "place", "location", "landmark", "park", "square", "street", "road", "playground", "beach", "forest", "desert", "city", "village", "station", "stop", "terminal", "airport", "harbor", "playground", "monument", "fountain", "bench", "parking_meter", "streetlight", "arena", "canyon", "cave", "harbor", "island", "landscape", "piazza", "region", "terrain", "valley"
    ],
    "Science & Technology": [
        "science", "technology", "lab", "experiment", "microscope", "telescope", "beaker", "test_tube", "flask", "centrifuge", "robotics", "drone", "battery", "sensor", "generator", "turbine", "engine", "gear", "circuit", "satellite", "syringe", "thermometer", "stethoscope", "scalpel", "bandage", "band_aid", "inhaler", "medicine", "pill", "atlas", "globe", "compass", "magnet", "binoculars", "bobbin", "bolt", "coil", "die", "drill", "fuse", "gear", "igniter", "ladder", "lightning_rod", "motor", "needle", "pendulum", "pliers", "power_shovel", "pulley", "ruler", "screwdriver", "stepladder", "tachometer", "timer", "tripod", "wrench", "apparatus", "component", "device", "evolution", "formula", "innovation", "mechanism"
    ],
    "Sports & Fitness": [
        "sport", "fitness", "ball", "bat", "racket", "club", "hockey_stick", "gym", "treadmill", "dumbbell", "barbell", "yoga_mat", "bicycle", "skateboard", "surfboard", "skis", "snowboard", "helmet", "goggles", "whistle", "trophy", "medal", "stadium", "field", "court", "swimming_pool", "fishing_rod", "hunting", "archery", "bow", "arrow", "dart", "frisbee", "ping_pong", "shuttlecock", "ski", "skate", "whistle", "baseball", "basketball", "beachball", "chess", "dartboard", "football", "golfcart", "hockey", "ping_pong", "scoreboard", "softball", "tennis", "volleyball", "athletics", "competition", "exercise", "fixture", "marathon", "tournament"
    ],
    "Weapons & Military": [
        "weapon", "gun", "rifle", "pistol", "canon", "missile", "tank", "armor", "shield", "sword", "knife", "dagger", "spear", "ax", "axe", "hatchet", "arrow", "bow", "explosive", "grenade", "mine", "barricade", "uniform", "helmet", "gasmask", "bulletproof_vest", "machine_gun", "projectile", "scabbard", "bayonet", "cannon", "rocket", "armory", "bayonet", "cannon", "grenade", "harpoon", "mace", "sling", "tank", "torpedo", "weapon", "ammunition", "artillery", "battleship", "brigade", "cavalry", "fortress", "garrison", "infantry", "ordnance", "trench", "military", "combat", "war", "tactical", "firearm"
    ]
}

def generate_taxonomy():
    with open('lvis_tag_list.json', 'r', encoding='utf-8') as f:
        tags = json.load(f)

    # 정의된 카테고리 및 확장 키워드
    categories = LVIS_TAXONOMY_DICT
    categorized_data = {cat: [] for cat in categories}
    uncategorized = []

    # 1. Primary Keyword Matching
    for tag in tags:
        tag_lower = tag.lower().replace('-', '_').replace(' ', '_')
        assigned = False
        
        for cat, keywords in categories.items():
            for kw in keywords:
                kw_clean = kw.lower().replace('-', '_').replace(' ', '_')
                if kw_clean == tag_lower or f"_{kw_clean}" in tag_lower or f"{kw_clean}_" in tag_lower:
                    categorized_data[cat].append(tag)
                    assigned = True
                    break
            if assigned: break
        
        if not assigned:
            uncategorized.append(tag)

    # 2. Secondary Logic for Uncategorized (Heuristics)
    remaining = []
    for tag in uncategorized:
        tag_lower = tag.lower()
        if any(x in tag_lower for x in ["cloth", "leather", "fabric", "silk"]):
            categorized_data["Fashion & Style"].append(tag)
        elif any(x in tag_lower for x in ["tool", "wrench", "hammer", "pliers", "saw", "measur"]):
            categorized_data["Science & Technology"].append(tag)
        elif any(x in tag_lower for x in ["storage", "holder", "rack", "container", "box", "case", "bag", "cover", "pot", "pan", "bottle", "jar", "keg"]):
            categorized_data["Furniture & Home"].append(tag)
        elif any(x in tag_lower for x in ["food", "drink", "edible", "cuisine", "fruit", "vegetable"]):
            categorized_data["Food & Drink"].append(tag)
        elif any(x in tag_lower for x in ["toy", "game", "lego"]):
            categorized_data["Art & Abstract"].append(tag)
        elif any(x in tag_lower for x in ["weapon", "pistol", "gun", "blade", "sharp"]):
            categorized_data["Weapons & Military"].append(tag)
        elif any(x in tag_lower for x in ["electronic", "circuit", "sensor", "digital"]):
            categorized_data["Electronics & Gadgets"].append(tag)
        else:
            remaining.append(tag)

    # Output JSON
    output = {
        "metadata": {
            "source": "LVIS Tag List",
            "categories_count": len(categories),
            "total_tags": len(tags),
            "categorized_tags": sum(len(v) for v in categorized_data.values()),
            "uncategorized_tags": len(remaining)
        },
        "taxonomy": categorized_data,
        "uncategorized": remaining
    }

    with open('lvis_taxonomy.json', 'w', encoding='utf-8') as outfile:
        json.dump(output, outfile, indent=4, ensure_ascii=False)
    
    print(f"Taxonomy saved to lvis_taxonomy.json")
    print(f"Categorized: {output['metadata']['categorized_tags']}, Uncategorized: {len(remaining)}")

if __name__ == "__main__":
    generate_taxonomy()
