# https://github.com/ultralytics/yolov3/blob/master/train.py

# Evolve hyperparameters
# Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
def evolve():
    meta = {
        "lr0": (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
        "lrf": (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
        "momentum": (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
        "weight_decay": (1, 0.0, 0.001),  # optimizer weight decay
        "warmup_epochs": (1, 0.0, 5.0),  # warmup epochs (fractions ok)
        "warmup_momentum": (1, 0.0, 0.95),  # warmup initial momentum
        "warmup_bias_lr": (1, 0.0, 0.2),  # warmup initial bias lr
        "box": (1, 0.02, 0.2),  # box loss gain
        "cls": (1, 0.2, 4.0),  # cls loss gain
        "cls_pw": (1, 0.5, 2.0),  # cls BCELoss positive_weight
        "obj": (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
        "obj_pw": (1, 0.5, 2.0),  # obj BCELoss positive_weight
        "iou_t": (0, 0.1, 0.7),  # IoU training threshold
        "anchor_t": (1, 2.0, 8.0),  # anchor-multiple threshold
        "anchors": (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
        "fl_gamma": (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
        "hsv_h": (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        "hsv_s": (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        "hsv_v": (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
        "degrees": (1, 0.0, 45.0),  # image rotation (+/- deg)
        "translate": (1, 0.0, 0.9),  # image translation (+/- fraction)
        "scale": (1, 0.0, 0.9),  # image scale (+/- gain)
        "shear": (1, 0.0, 10.0),  # image shear (+/- deg)
        "perspective": (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        "flipud": (1, 0.0, 1.0),  # image flip up-down (probability)
        "fliplr": (0, 0.0, 1.0),  # image flip left-right (probability)
        "mosaic": (1, 0.0, 1.0),  # image mixup (probability)
        "mixup": (1, 0.0, 1.0),  # image mixup (probability)
        "copy_paste": (1, 0.0, 1.0),
    }  # segment copy-paste (probability)

    with open(opt.hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)  # load hyps dict
        if "anchors" not in hyp:  # anchors commented in hyp.yaml
            hyp["anchors"] = 3
    if opt.noautoanchor:
        del hyp["anchors"], meta["anchors"]
    opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
    # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
    evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
    if opt.bucket:
        # download evolve.csv if exists
        subprocess.run(
            [
                "gsutil",
                "cp",
                f"gs://{opt.bucket}/evolve.csv",
                str(evolve_csv),
            ]
        )

    for _ in range(opt.evolve):  # generations to evolve
        if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
            # Select parent(s)
            parent = "single"  # parent selection method: 'single' or 'weighted'
            x = np.loadtxt(evolve_csv, ndmin=2, delimiter=",", skiprows=1)
            n = min(5, len(x))  # number of previous results to consider
            x = x[np.argsort(-fitness(x))][:n]  # top n mutations
            w = fitness(x) - fitness(x).min() + 1e-6  # weights (sum > 0)
            if parent == "single" or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # random selection
                x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
            elif parent == "weighted":
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

            # Mutate
            mp, s = 0.8, 0.2  # mutation probability, sigma
            npr = np.random
            npr.seed(int(time.time()))
            g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
            ng = len(meta)
            v = np.ones(ng)
            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
            for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                hyp[k] = float(x[i + 7] * v[i])  # mutate

        # Constrain to limits
        for k, v in meta.items():
            hyp[k] = max(hyp[k], v[1])  # lower limit
            hyp[k] = min(hyp[k], v[2])  # upper limit
            hyp[k] = round(hyp[k], 5)  # significant digits

        # Train mutation
        results = train(hyp.copy(), opt, device, callbacks)
        callbacks = Callbacks()
        # Write mutation results
        keys = (
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP_0.5",
            "metrics/mAP_0.5:0.95",
            "val/box_loss",
            "val/obj_loss",
            "val/cls_loss",
        )
        print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

    # Plot results
    plot_evolve(evolve_csv)
    LOGGER.info(
        f'Hyperparameter evolution finished {opt.evolve} generations\n'
        f"Results saved to {colorstr('bold', save_dir)}\n"
        f'Usage example: $ python train.py --hyp {evolve_yaml}'
)
