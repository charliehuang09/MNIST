from tqdm import trange
import train
def main():
    runs = 10
    for run in trange(runs):
        train.main(run_name=f"runs/R{run}")

if __name__ == "__main__":
    main()
