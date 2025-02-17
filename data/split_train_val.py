import time
import argparse
import pprint as pp
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=200)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--node_dim", type=int, default=2)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/mnt/home/zuwang/workspace/graph-convnet-tsp/data/") 
    opts = parser.parse_args()
    
    if opts.filename is None:
        opts.filename = f"tsp{opts.num_nodes}_concorde.txt"
    opts.filename = os.path.join(opts.output_dir, opts.filename)
    
    # Pretty print the run args
    pp.pprint(vars(opts))
    
    start_time = time.time()
    
    filedata = open(opts.filename, "r").readlines()
    print("Total samples: ", len(filedata))
    val_data = filedata[:opts.val_size]
    print("Validation samples: ", len(val_data))
    train_data = filedata[opts.val_size:]
    print("Training samples: ", len(train_data))
    
    # Create separate validation data file
    with open(opts.output_dir + "tsp{}_val_concorde.txt".format(opts.num_nodes), "w", encoding="utf-8") as f:
        for line in val_data:
            f.write(line)
    
    # Create separate train data file
    with open(opts.output_dir + "tsp{}_train_concorde.txt".format(opts.num_nodes), "w", encoding="utf-8") as f:
        for line in train_data:
            f.write(line)
    
    end_time = time.time() - start_time
    
    print(f"Total time: {end_time/3600:.1f}")
    