import io
import os
import sys
import contextlib
import torch
import torchvision
from torchinfo import summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet18(weights=None).to(device)
    model.eval()

    buf = io.StringIO()
    # Some torchinfo versions forward unexpected kwargs to the model call.
    # To avoid that, capture stdout while calling summary instead of using print_fn.
    with contextlib.redirect_stdout(buf):
        summary(
            model,
            input_size=(1, 3, 224, 224),
            col_names=("output_size", "num_params"),
            verbose=1,
        )

    out = buf.getvalue()
    out_path = os.path.join(os.path.dirname(__file__), "resnet18_profile.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)

    print(f"Wrote torchinfo profile to {out_path}")


if __name__ == "__main__":
    main()
