# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:22:37 2019

@author: Gareth
"""


def test_LSTMmain_initial():
    """Test of differentiability of LSTMmain - direct integration test with LSTMunit
    """
    shape = [2, 4, 1, 16, 16]
    test_input_tensor = torch.zeros(shape, dtype=torch.double, requires_grad=True)

    test2 = LSTMmain_t(
        shape, 1, 3, 5, [1], test_input=[1, 2], copy_bool=[False, False], debug=False
    ).double()

    #    ans, _ = test2(test_input_tensor, copy_in = [False,False], copy_out = [False, False])
    # internal outputs - list. Can we switch to tensors.
    res = torch.autograd.gradcheck(
        test2, (test_input_tensor,), eps=1e-4, raise_exception=True
    )


def test_LSTMunit_autograd():
    """Tests end to end differentiability of LSTMunit_t.
    """
    shape = [1, 1, 16, 16]
    x = torch.zeros(shape, dtype=torch.double, requires_grad=True)
    h = torch.zeros([1, 2, 16, 16], dtype=torch.double, requires_grad=True)
    c = torch.zeros([1, 2, 16, 16], dtype=torch.double, requires_grad=True)
    testunit = LSTMunit_t(1, 2, 3).double()
    torch.autograd.gradcheck(testunit, (x, h, c), eps=1e-4, raise_exception=True)


def test_LSTMencdec():
    """Tests end to end differentiability of LSTMencdec
    """
    structure = np.array([[2, 4, 0], [0, 4, 2]])
    encdec = LSTMencdec_onestep_t(structure, 1, 5)
    shape = [1, 10, 1, 16, 16]
    x = torch.zeros(shape, dtype=torch.double, requires_grad=True)
    torch.autograd.gradcheck(encdec, (x,), eps=1e-4, raise_exception=True)


# test_LSTMmain_initial()
# test_LSTMencdec()
