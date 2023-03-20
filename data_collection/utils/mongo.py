import logging
import pymongo.collection
from pymongo.errors import BulkWriteError, DuplicateKeyError, WriteError

logger = logging.getLogger(__name__)


def aggr_one(self, pipeline):
    cursor = self.aggregate(pipeline)
    return next(cursor, None)


pymongo.collection.Collection.aggregate_one = aggr_one


def match_in_collection(collection, condition):
    return len(list(collection.find(condition))) > 0


def duplicate_key_error_handler(insert_fn):
    """
    Handles DuplicateKeyError commonly thrown by pymongo collection inserts. Can be used as a decorator.

    Args:
        insert_fn: The function that throws these exceptions.
    """

    def inner_fn(*args, **kwargs):
        try:
            return insert_fn(*args, **kwargs)
        except DuplicateKeyError as e:
            logger.debug(e.details)
        except BulkWriteError as e:
            for write_error in e.details["writeErrors"]:
                # just log if duplicate key error occurs
                if write_error["code"] == 11000:
                    logger.debug(write_error)
                else:
                    raise WriteError(
                        error=write_error["errmsg"],
                        code=write_error["code"],
                    )

    return inner_fn


@duplicate_key_error_handler
def insert_many(collection, items):
    if len(items) > 0:
        insert_result = collection.insert_many(items)
        logger.debug(
            "Inserted %d items into %s collection."
            % (len(insert_result.inserted_ids), collection.name)
        )
    else:
        logger.warn("No items in list to insert into %s." % collection.name)


@duplicate_key_error_handler
def insert_one(collection, item):
    if item is not None:
        collection.insert_one(item)
    else:
        logger.warn("Item is None, can't insert into %s." % collection.name)
