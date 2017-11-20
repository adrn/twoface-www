from os import path

from twoface.log import log as logger
from twoface.config import TWOFACE_CACHE_PATH

from twoface_www import app

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0,
                          dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0,
                          dest='quietness')

    parser.add_argument('--debug', dest='debug', action='store_true',
                        default=False, help='Enable debug mode.')

    parser.add_argument('--cache', dest='cache_path',
                        default=TWOFACE_CACHE_PATH,
                        type=str, help='Path to the twoface cache.')
    parser.add_argument('--db', dest='db_file',
                        default='apogee.sqlite',
                        type=str, help='Name of the database file.')

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbosity != 0:
        if args.verbosity == 1:
            logger.setLevel(logging.DEBUG)
        else: # anything >= 2
            logger.setLevel(1)

    elif args.quietness != 0:
        if args.quietness == 1:
            logger.setLevel(logging.WARNING)
        else: # anything >= 2
            logger.setLevel(logging.ERROR)

    else: # default
        logger.setLevel(logging.INFO)

    app.config.update(dict(
        DATABASE_PATH=path.join(args.cache_path, args.db_file),
        SECRET_KEY=b',\xdfnL\x13H\x80R\x96B\xb1\xe2.\x1a\x8c\xae;\x1a \x97;@\x03\xec'
    ))

    app.run(debug=args.debug)
