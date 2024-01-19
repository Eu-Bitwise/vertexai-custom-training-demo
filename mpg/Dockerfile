FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3
WORKDIR /root

WORKDIR /

# Copy the trainer code to the docker image.
COPY trainer /trainer

# Entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]