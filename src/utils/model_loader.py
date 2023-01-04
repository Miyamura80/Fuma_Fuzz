from models import PPO

def get_model(args):

    if args.model == "PPO":
        
        model = PPO(
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=device,
            scatter=args.scatter,
            drpt_prob=args.dropout,
        )
        return model


