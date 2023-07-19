import torch
from torchvision import transforms
from PIL import Image

labels = {0: 'airfield', 1: 'airplane cabin', 2: 'airport terminal', 3: 'alcove', 4: 'alley', 5: 'amphitheater', 6: 'amusement arcade',
 7: 'amusement park', 8: 'apartment building outdoor', 9: 'aquarium', 10: 'aqueduct', 11: 'arcade', 12: 'arch', 13: 'archaelogical excavation',
 14: 'archive', 15: 'arena hockey', 16: 'arena performance', 17: 'arena rodeo', 18: 'army base', 19: 'art gallery', 20: 'art school',
 21: 'art studio', 22: 'artists loft', 23: 'assembly line', 24: 'athletic field outdoor', 25: 'atrium public', 26: 'attic',
 27: 'auditorium', 28: 'auto factory', 29: 'auto showroom', 30: 'badlands', 31: 'bakery shop', 32: 'balcony exterior', 33: 'balcony interior',
 34: 'ball pit', 35: 'ballroom', 36: 'bamboo forest', 37: 'bank vault', 38: 'banquet hall', 39: 'bar', 40: 'barn', 41: 'barndoor',
 42: 'baseball field', 43: 'basement', 44: 'basketball court indoor', 45: 'bathroom', 46: 'bazaar indoor', 47: 'bazaar outdoor', 48: 'beach', 49: 'beach house',
 50: 'beauty salon', 51: 'bedchamber', 52: 'bedroom', 53: 'beer garden', 54: 'beer hall', 55: 'berth', 56: 'biology laboratory',
 57: 'boardwalk', 58: 'boat deck', 59: 'boathouse', 60: 'bookstore', 61: 'booth indoor', 62: 'botanical garden', 63: 'bow window indoor',
 64: 'bowling alley', 65: 'boxing ring', 66: 'bridge', 67: 'building facade', 68: 'bullring', 69: 'burial chamber', 70: 'bus interior',
 71: 'bus station indoor', 72: 'butchers shop', 73: 'butte', 74: 'cabin outdoor', 75: 'cafeteria', 76: 'campsite', 77: 'campus',
 78: 'canal natural', 79: 'canal urban', 80: 'candy store', 81: 'canyon', 82: 'car interior', 83: 'carousel', 84: 'castle',
 85: 'catacomb', 86: 'cemetery', 87: 'chalet', 88: 'chemistry lab', 89: 'childs room', 90: 'church indoor', 91: 'church outdoor',
 92: 'classroom', 93: 'clean room', 94: 'cliff', 95: 'closet', 96: 'clothing store', 97: 'coast', 98: 'cockpit', 99: 'coffee shop',
 100: 'computer room', 101: 'conference center', 102: 'conference room', 103: 'construction site', 104: 'corn field', 105: 'corral',
 106: 'corridor', 107: 'cottage', 108: 'courthouse', 109: 'courtyard', 110: 'creek', 111: 'crevasse', 112: 'crosswalk', 113: 'dam',
 114: 'delicatessen', 115: 'department store', 116: 'desert sand', 117: 'desert vegetation', 118: 'desert road', 119: 'diner outdoor',
 120: 'dining hall', 121: 'dining room', 122: 'discotheque', 123: 'doorway outdoor', 124: 'dorm room', 125: 'downtown', 126: 'dressing room',
 127: 'driveway', 128: 'drugstore', 129: 'elevator door', 130: 'elevator lobby', 131: 'elevator shaft', 132: 'embassy', 133: 'engine room',
 134: 'entrance hall', 135: 'escalator indoor', 136: 'excavation', 137: 'fabric store', 138: 'farm', 139: 'fastfood restaurant',
 140: 'field cultivated', 141: 'field wild', 142: 'field road', 143: 'fire escape', 144: 'fire station', 145: 'fishpond',
 146: 'flea market indoor', 147: 'florist shop indoor', 148: 'food court', 149: 'football field', 150: 'forest broadleaf',
 151: 'forest path', 152: 'forest road', 153: 'formal garden', 154: 'fountain', 155: 'galley', 156: 'garage indoor', 157: 'garage outdoor',
 158: 'gas station', 159: 'gazebo exterior', 160: 'general store indoor', 161: 'general store outdoor', 162: 'gift shop', 163: 'glacier',
 164: 'golf course', 165: 'greenhouse indoor', 166: 'greenhouse outdoor', 167: 'grotto', 168: 'gymnasium indoor', 169: 'hangar indoor',
 170: 'hangar outdoor', 171: 'harbor', 172: 'hardware store', 173: 'hayfield', 174: 'heliport', 175: 'highway', 176: 'home office',
 177: 'home theater', 178: 'hospital', 179: 'hospital room', 180: 'hot spring', 181: 'hotel outdoor', 182: 'hotel room', 183: 'house',
 184: 'hunting lodge outdoor', 185: 'ice cream parlor', 186: 'ice floe', 187: 'ice shelf', 188: 'ice skating rink indoor', 189: 'ice skating rink outdoor',
 190: 'iceberg', 191: 'igloo', 192: 'industrial area', 193: 'inn outdoor', 194: 'islet', 195: 'jacuzzi indoor', 196: 'jail cell',
 197: 'japanese garden', 198: 'jewelry shop', 199: 'junkyard', 200: 'kasbah', 201: 'kennel outdoor', 202: 'kindergarden classroom',
 203: 'kitchen', 204: 'lagoon', 205: 'lake natural', 206: 'landfill', 207: 'landing deck', 208: 'laundromat', 209: 'lawn',
 210: 'lecture room', 211: 'legislative chamber', 212: 'library indoor', 213: 'library outdoor', 214: 'lighthouse',
 215: 'living room', 216: 'loading dock', 217: 'lobby', 218: 'lock chamber', 219: 'locker room', 220: 'mansion', 221: 'manufactured home',
 222: 'market indoor', 223: 'market outdoor', 224: 'marsh', 225: 'martial arts gym', 226: 'mausoleum', 227: 'medina', 228: 'mezzanine',
 229: 'moat water', 230: 'mosque outdoor', 231: 'motel', 232: 'mountain', 233: 'mountain path', 234: 'mountain snowy', 235: 'movie theater indoor',
 236: 'museum indoor', 237: 'museum outdoor', 238: 'music studio', 239: 'natural history museum', 240: 'nursery', 241: 'nursing home',
 242: 'oast house', 243: 'ocean', 244: 'office', 245: 'office building', 246: 'office cubicles', 247: 'oilrig', 248: 'operating room',
 249: 'orchard', 250: 'orchestra pit', 251: 'pagoda', 252: 'palace', 253: 'pantry', 254: 'park', 255: 'parking garage indoor',
 256: 'parking garage outdoor', 257: 'parking lot', 258: 'pasture', 259: 'patio', 260: 'pavilion', 261: 'pet shop', 262: 'pharmacy',
 263: 'phone booth', 264: 'physics laboratory', 265: 'picnic area', 266: 'pier', 267: 'pizzeria', 268: 'playground', 269: 'playroom',
 270: 'plaza', 271: 'pond', 272: 'porch', 273: 'promenade', 274: 'pub indoor', 275: 'racecourse', 276: 'raceway', 277: 'raft',
 278: 'railroad track', 279: 'rainforest', 280: 'reception', 281: 'recreation room', 282: 'repair shop', 283: 'residential neighborhood',
 284: 'restaurant', 285: 'restaurant kitchen', 286: 'restaurant patio', 287: 'rice paddy', 288: 'river', 289: 'rock arch',
 290: 'roof garden', 291: 'rope bridge', 292: 'ruin', 293: 'runway', 294: 'sandbox', 295: 'sauna', 296: 'schoolhouse',
 297: 'science museum', 298: 'server room', 299: 'shed', 300: 'shoe shop', 301: 'shopfront', 302: 'shopping mall indoor',
 303: 'shower', 304: 'ski resort', 305: 'ski slope', 306: 'sky', 307: 'skyscraper', 308: 'slum', 309: 'snowfield', 310: 'soccer field',
 311: 'stable', 312: 'stadium baseball', 313: 'stadium football', 314: 'stadium soccer', 315: 'stage indoor', 316: 'stage outdoor',
 317: 'staircase', 318: 'storage room', 319: 'street', 320: 'subway station platform', 321: 'supermarket', 322: 'sushi bar',
 323: 'swamp', 324: 'swimming hole', 325: 'swimming pool indoor',326: 'swimming pool outdoor', 327: 'synagogue outdoor', 328: 'television room',
 329: 'television studio', 330: 'temple asia', 331: 'throne room', 332: 'ticket booth', 333: 'topiary garden', 334: 'tower', 335: 'toyshop',
 336: 'train interior', 337: 'train station platform', 338: 'tree farm', 339: 'tree house', 340: 'trench', 341: 'tundra', 342: 'underwater ocean deep',
 343: 'utility room', 344: 'valley', 345: 'vegetable garden', 346: 'veterinarians office', 347: 'viaduct', 348: 'village', 349: 'vineyard',
 350: 'volcano', 351: 'volleyball court outdoor', 352: 'waiting room', 353: 'water park', 354: 'water tower', 355: 'waterfall',
 356: 'watering hole', 357: 'wave', 358: 'wet bar', 359: 'wheat field', 360: 'wind farm', 361: 'windmill', 362: 'yard', 363: 'youth hostel',
 364: 'zen garden'}

image_prep = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

scene_model = torch.load('densenet_asgd_best.pt', map_location=device)


def predict_scene(model, transforms, image, labels):
#    image = Image.open(image_path)
    prepped_image = transforms(image)
    img_tensor = torch.unsqueeze(prepped_image, 0)
    model = model.to(device)
    img_tensor = img_tensor.to(device)
    model.eval()
    prediction = model(img_tensor)
    top1 = torch.topk(prediction, 1)
    label = labels[top1[1].item()]

    return label

def run_densenet(raw_output, ilabels, image):
    predicted_label = predict_scene(scene_model, image_prep, image, labels)
    ilabels.append('seen at '+predicted_label)
    raw_output.append({'scene': predicted_label})

    return ilabels, raw_output