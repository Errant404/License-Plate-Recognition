from detect副本 import parse_opt

opt, parser = parse_opt()


parser.set_defaults(save_txt=True)

# opt.set_defaults('save-txt'=True)
opt = parser.parse_args()
print(opt)
