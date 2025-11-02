langs = cs de el es it ms nl pl pt ru tr

process_all:
	$(foreach l,$(langs),rm ./data/$(l)/$(l)_*/$(l)_*.wlh;)
	$(foreach l,$(langs),python ./scripts/process_dump.py --lang $(l);)

translate_all:
	$(foreach l,$(langs),rm ./data/$(l)/$(l)_*/$(l)_*.tra;)
	$(foreach l,$(langs),python ./scripts/make_tr.py ./data/$(l)/$(l)_*/$(l)_*;)