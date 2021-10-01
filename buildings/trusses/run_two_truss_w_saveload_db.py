import o3seespy as o3
import numpy as np
import os
import o3plot
import matplotlib.pyplot as plt
import hashlib
import json


def _json_default(o):
    """Converts numpy types to json serialisable python types"""
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def compute_unique_hash(obj_dict):
    return hashlib.md5(json.dumps(obj_dict).encode('utf-8')).hexdigest()


def collect_serial_value(value, all_objs):
    """
    Introspective function that returns a serialisable value

    The function converts objects to dictionaries
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, int):
        return value
    elif isinstance(value, np.int64):
        return int(value)
    elif hasattr(value, "__len__"):
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = value.tolist()
        values = []
        for item in value:
            if hasattr(item, 'op_type'):
                obj_dict = convert2dict(item, all_objs)
                uhash = compute_unique_hash(obj_dict)
                if uhash not in all_objs:
                    all_objs[uhash] = obj_dict
                val = 'unique_hash_' + uhash
                values.append(val)
            else:
                values.append(collect_serial_value(item, all_objs))
        return values
    else:
        return value


def convert2dict(obj, all_objs):
    # TODO: do not export None
    items = dir(obj)
    dd = {}
    for item in items:
        if item.startswith('_'):
            continue
        val = getattr(obj, item)
        name = item
        if item == 'osi':
            continue
        elif hasattr(val, 'op_type'):
            obj_dict = convert2dict(val, all_objs)
            uhash = compute_unique_hash(obj_dict)
            if uhash not in all_objs:
                all_objs[uhash] = obj_dict
            val = uhash
            name = 'unique_hash_' + item

        if callable(val):
            continue
        else:
            val = collect_serial_value(val, all_objs)
        dd[name] = val
    return dd


class O3ObjectDatabase(object):
    def __init__(self, ffp=None):
        self.objs = {}
        self.name2hash = {}
        if ffp is None:
            self.ffp = None
        else:
            self.ffp = f'{ffp}.json'

    def add_obj(self, obj, oname):
        obj_dict = convert2dict(obj, self.objs)
        uhash = compute_unique_hash(obj_dict)
        if uhash not in self.objs:
            self.objs[uhash] = obj_dict
        self.name2hash[oname] = uhash

    def add_objs(self, objs, oname):
        all_dds = []
        for obj in objs:
            obj_dict = convert2dict(obj, self.objs)
            uhash = compute_unique_hash(obj_dict)
            if uhash not in self.objs:
                self.objs[uhash] = obj_dict
            all_dds.append(f'unique_hash_{uhash}')
        self.name2hash[oname] = all_dds

    def to_file(self, ffp, indent=4):

        od = {'name2hash': self.name2hash, 'objs': self.objs}
        json.dump(od, open(ffp, "w"), indent=indent, default=_json_default)


def get_key_value(val, hash2objs, key=None):
    if key is not None and key.startswith('unique_hash_'):
        child_obj = hash2objs[val]
        return key.replace('unique_hash_', ''), child_obj
    if val is None:
        return key, val
    if isinstance(val, str):
        if val.startswith('unique_hash_'):
            child_obj = hash2objs[val.replace('unique_hash_', '')]
            return key, child_obj
        else:
            return key, val
    elif isinstance(val, list):
        vals = []
        for item in val:
            ikey, val = get_key_value(item, hash2objs)
            vals.append(val)
        return key, vals
    elif isinstance(val, dict):
        vals = {}
        for item in val:
            ikey, ivalue = get_key_value(val[item], hash2objs, key=item)
            vals[ikey] = ivalue
        return key, vals
    else:
        return key, val



def add_to_obj(obj, dictionary, hash2objs=None, exclusions=None, verbose=0):
    """
    Cycles through a dictionary and adds the key-value pairs to an object.

    Parameters
    ----------
    obj: object
        An object that parameters should be added to
    dictionary: dict
        Keys are object parameter names, values are object parameter values
    exceptions: list
        Parameters that should be excluded
    verbose: bool
        If true then show print statements
    :return:
    """
    if exclusions is None:
        exclusions = []
    # exceptions.append('unique_hash')
    for item in dictionary:
        val = dictionary[item]
        if item in exclusions:
            continue
        print(item, val)
        key, value = get_key_value(val, hash2objs, key=item)
        setattr(obj, key, value)


def connect_to_db_and_obj_holder(ffp):
    o3.ops.database('File', ffp)
    return O3ObjectDatabase(ffp=ffp)


def save_db_state(state_tag):
    o3.ops.save(state_tag)

def save_db_state_and_objs(state_tag, o3db):
    o3db.to_file(o3db.ffp)
    o3.ops.save(state_tag)


def restore_db_state(state_tag):
    o3.ops.restore(state_tag)


def restore_objs2dict(ffp):
    if not ffp.endswith('.json'):
        ffp = f'{ffp}.json'
    data = json.load(open(ffp))

    hash2obj = {}
    for uhash in data['objs']:
        hash2obj[uhash] = RestoredObj()
    for uhash in data['objs']:
        val = data['objs'][uhash]
        obj = hash2obj[uhash]
        add_to_obj(obj, val, hash2obj)

    dd = {}
    for name in data['name2hash']:
        val = data['name2hash'][name]
        if isinstance(val, str):
            dd[name] = hash2obj[val]
        else:
            vals = []
            for item in val:
                uhash = item.replace('unique_hash_', '')
                vals.append(hash2obj[uhash])
            dd[name] = vals
    return dd


class RestoredObj(object):

    def __init__(self, tag=None):
        self.tag = tag


def run(out_folder):

    osi = o3.OpenSeesInstance(ndm=2, ndf=2, state=3)
    x_centre = 0.0
    y_centre = 0.0
    top_node = o3.node.Node(osi, x_centre, y_centre)
    o3db = connect_to_db_and_obj_holder('db/truss_ops')
    # o3db = O3ObjectDatabase()
    o3db.add_obj(top_node, 'top_node')
    bot_nodes = []
    sf_eles = []

    o3.Mass(osi, top_node, 10, 10)

    fy = 500
    k = 1.0e4
    b = 0.1
    pro_params = [5, 0.925, 0.15]
    # sf_mat = o3.uniaxial_material.SteelMPF(osi, fy, fy, k, b, b, params=pro_params)
    sf_mat = o3.uniaxial_material.Elastic(osi, k)

    diff_pos = 0.5
    depth = 1
    bot_nodes.append(o3.node.Node(osi, x_centre - diff_pos, y_centre - depth))
    o3.Fix2DOF(osi, bot_nodes[0], o3.cc.FIXED, o3.cc.FIXED)
    bot_nodes.append(o3.node.Node(osi, x_centre + diff_pos, y_centre - depth))
    o3.Fix2DOF(osi, bot_nodes[1], o3.cc.FIXED, o3.cc.FIXED)

    sf_eles.append(o3.element.Truss(osi, [top_node, bot_nodes[0]], big_a=1.0, mat=sf_mat))
    sf_eles.append(o3.element.Truss(osi, [top_node, bot_nodes[1]], big_a=1.0, mat=sf_mat))
    o3db.add_objs(sf_eles, 'sf_eles')
    save_db_state_and_objs(1, o3db)
    # o3.ops.database('File', 'db/truss_ops')
    # o3.ops.save(1)
    # o3db.to_file('db/truss_ops.json')


def restart():
    ffp = 'db/truss_ops'
    osi = o3.OpenSeesInstance(3, restore=(ffp, 1))
    # o3.ops.database('File', 'db/truss_ops')
    # o3.ops.restore(1)
    dd = restore_objs2dict(ffp)
    top_node = dd['top_node']
    sf_eles = dd['sf_eles']
    ts0 = o3.time_series.Linear(osi, factor=1)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, top_node, [100, -500])

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.FullGeneral(osi)
    n_steps_gravity = 15
    d_gravity = 1. / n_steps_gravity
    o3.integrator.LoadControl(osi, d_gravity, num_iter=10)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)
    o3r = o3.results.Results2D(cache_path=out_folder, dynamic=True)
    o3r.pseudo_dt = 0.1
    o3r.start_recorders(osi, dt=0.1)
    nr = o3.recorder.NodeToArrayCache(osi, top_node, [o3.cc.DOF2D_X, o3.cc.DOF2D_Y], 'disp')
    er = o3.recorder.ElementToArrayCache(osi, sf_eles[0], arg_vals=['force'])
    for i in range(n_steps_gravity):
        o3.analyze(osi, num_inc=1)
    o3.load_constant(osi, time=0.0)
    import o3seespy.extensions
    o3.extensions.to_py_file(osi, 'ofile.py')
    print('init_disp: ', o3.get_node_disp(osi, top_node, o3.cc.DOF2D_Y))
    print('init_disp: ', o3.get_node_disp(osi, top_node, o3.cc.DOF2D_Y))
    print('init_disp: ', o3.get_node_disp(osi, top_node, o3.cc.DOF2D_Y))
    o3.wipe(osi)
    o3r.save_to_cache()
    # o3r.coords = o3.get_all_node_coords(osi)
    # o3r.ele2node_tags = o3.get_all_ele_node_tags_as_dict(osi)
    data = nr.collect()
    edata = er.collect()
    # bf, sps = plt.subplots(nrows=2)
    # sps[0].plot(data[:, 0])
    # sps[0].plot(data[:, 1])
    # sps[1].plot(edata[:, 0])
    # # sps[0].plot(data[1])
    # plt.show()
    o3r.load_from_cache()
    o3plot.replot(o3r)



if __name__ == '__main__':
    name = __file__.replace('.py', '')
    name = name.split("run_")[-1]
    import all_paths as ap

    out_folder = ap.OP_PATH + name + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    run(out_folder)
    restart()